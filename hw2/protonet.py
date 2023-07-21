"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

import omniglot
import util
from collections import defaultdict

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """

        self._network = ProtoNetNetwork()
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # For a given task, compute the prototypes and the protonet loss.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate loss_batch, accuracy_support_batch, and
            # accuracy_query_batch.

            # Step 1. Get the embeddings for images in the support and query sets
            images_support_emb = self._network(images_support)
            images_query_emb = self._network(images_query)

            # Step 2. Get the prototype vectors corresponding to support set images
            prototype_dict = defaultdict(list)
            for idx, label_support in enumerate(labels_support):
                prototype_dict[int(label_support)].append(images_support_emb[idx])

            prototype_dict = {k: torch.stack(v).mean(dim=0) for k, v in prototype_dict.items()}

            images_support_prototype_emb = torch.stack(list(prototype_dict.values()))
            labels_support_prototype_emb = torch.tensor(list(prototype_dict.keys()))

            # Step 3. Calculate euclidian distances
            dist_support_batch = torch.cdist(images_support_emb, images_support_prototype_emb, p=2)
            dist_query_batch = torch.cdist(images_query_emb, images_support_prototype_emb, p=2)

            # Step 4. Calculate the losses
            ## Get targets in the order of prototype labels
            target_support = torch.gather(labels_support_prototype_emb, 0, labels_support)
            target_query = torch.gather(labels_support_prototype_emb, 0, labels_query)

            ## Calculate cross entropy loss over the query set
            loss_batch.append(F.cross_entropy(input=-dist_query_batch, target=target_query))

            ## Calculate accuracy
            accuracy_support_batch.append(util.score(-dist_support_batch, target_support))
            accuracy_query_batch.append(util.score(-dist_query_batch, target_query))

            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=5,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
