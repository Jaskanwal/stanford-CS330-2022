"""
Factorization models for implicit feedback problems.
"""
import losses
import torch
import utils

import numpy as np
import torch.optim as optim

from models import MultiTaskNet


class MultitaskModel(object):
    """
    A multitask model with implicit feedback matrix factorization
    and MLP regression. Uses a classic matrix factorization [1]_
    approach, with latent vectors used to represent both users
    and items. Their dot product gives the predicted interaction
    probability for a user-item pair. The predicted numerical
    score is obtained by processing the user and item representation
    through an MLP network [2]_.

    The factorization loss is constructed through negative sampling: 
    for any known user-item pair, one or more items are randomly
    sampled to act as negatives (expressing a lack of preference
    by the user for the sampled item). The regression training is
    structured as standard supervised learning.

    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).

    .. [2] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie,
           Xia Hu, and Tat-Seng Chua. "Neural collaborative filtering."
           In Proceedings of the 26th international conference on
           worldwide web, pages 173–182, (2017).
    Parameters
    ----------

    interactions: class:Interactions
        Dataset of user-item interactions.
    factorization_weight: float, optional
        Weight for factorization loss.
    regression_weight: float, optional
        Weight for regression loss.
    embedding_dim: int, optional
        Number of embedding dimensions to use for users and items.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    representation: a representation module, optional
        If supplied, will override default settings and be used as the
        main network module in the model. Intended to be used as an escape
        hatch when you want to reuse the model's training functions but
        want full freedom to specify your network topology.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.

    """

    def __init__(self,
                 interactions,
                 factorization_weight = 0.5,
                 regression_weight = 0.5,
                 embedding_dim=32,
                 n_iter=1,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-3,
                 optimizer_func=None,
                 use_cuda=False,
                 representation=None,
                 sparse=False,
                 random_state=None):

        self._factorization_weight = factorization_weight
        self._regression_weight = regression_weight
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._representation = representation
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        if self._representation is not None:
            self._net = utils.gpu(self._representation,
                                  self._use_cuda)
        else:
            self._net = utils.gpu(
                MultiTaskNet(self._num_users,
                             self._num_items,
                             self._embedding_dim,
                             sparse=self._sparse),
                self._use_cuda
            )

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        self._factorization_loss_func = losses.bpr_loss
        self._regression_loss_func = losses.regression_loss


    def _check_input(self, user_ids, item_ids, allow_items_none=False):
        """
        Verify input data is valid and arise corresponding error otherwise.

        Parameters
        ----------

        user_ids: array
            An array of integer user IDs of shape (batch,)
        item_ids: array or None
            An array of integer item IDs of shape (batch,)

        """

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: class:Interactions
            The input dataset.

        Returns
        -------

        factorization_loss: float
            Mean factorization loss over the epoch.
   
        regression_loss: float
            Mean regression loss over the epoch.
    
        epoch_loss: float
            Joint weighted model loss over the epoch.
        """

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        self._check_input(user_ids, item_ids)

        for _ in range(self._n_iter):

            users, items, ratings = utils.shuffle([user_ids,
                                                   item_ids,
                                                   interactions.ratings],
                                                   random_state=self._random_state)

            user_ids_tensor = utils.gpu(torch.from_numpy(users),
                                        self._use_cuda)
            item_ids_tensor = utils.gpu(torch.from_numpy(items),
                                        self._use_cuda)
            ratings_tensor = utils.gpu(torch.from_numpy(ratings),
                                       self._use_cuda)

            epoch_factorization_loss = []
            epoch_regression_loss = []
            epoch_loss = []

            for (batch_user,
                 batch_item,
                 batch_ratings) in utils.minibatch([user_ids_tensor,
                                                    item_ids_tensor,
                                                    ratings_tensor],
                                                    batch_size=self._batch_size):

                positive_prediction, score = self._net(batch_user, batch_item)
                negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                #Losses
                factorization_loss = self._factorization_loss_func(
                                   positive_prediction,
                                   negative_prediction)
                epoch_factorization_loss.append(factorization_loss.item())

                regression_loss = self._regression_loss_func(batch_ratings,
                                                             score)
                epoch_regression_loss.append(regression_loss.item())

                loss = (self._factorization_weight * factorization_loss +
                       self._regression_weight * regression_loss)
                epoch_loss.append(loss.item())

                loss.backward()
                self._optimizer.step()


        return (np.mean(epoch_factorization_loss), 
                np.mean(epoch_regression_loss),
                np.mean(epoch_loss))

        
    def _get_negative_prediction(self, user_ids):
        """
        Generate negative predictions for user-item interactions, 
        corresponds to p_ij^- in the assignment.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)


        Returns
        -------

        negative_prediction: tensor
            A tensor of user-item interaction log-probability
            of shape (batch,)
        """

        negative_items = self._random_state.randint(0, self._num_items,
                                                    len(user_ids),
                                                    dtype=np.int64)
        negative_var = utils.gpu(torch.from_numpy(negative_items),
                                 self._use_cuda)
        negative_prediction, _ = self._net(user_ids, negative_var)

        return negative_prediction


    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        user_ids, item_ids = utils.process_ids(user_ids, item_ids,
                                               self._num_items, self._use_cuda)

        positive_prediction, score = self._net(user_ids, item_ids)

        return utils.cpu(positive_prediction).detach().numpy().flatten(), \
               utils.cpu(score).detach().numpy().flatten()
