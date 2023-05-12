import torch
import numpy as np


def gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(tensors, batch_size=128):
    """
    A generator object that yields minibatches from a list of tensors

    Parameters
    ----------

    tensors: list
        A list of tensors, with the same length

    batch_size: int
        Size of the batches to be returnes
    """
    for i in range(0, len(tensors[0]), batch_size):
        yield [x[i:i + batch_size] for x in tensors]


def process_ids(user_ids, item_ids, num_items, use_cuda):
    """
    Process user_ids and provide all item_ids if 
    they have not been supplied

    Parameters
    ----------

    user_ids: int or array
        An integer or an array of size (num_items,)

    item_ids: array or None
        An array of size (num_items, ) or None. If None
        items IDs will be supplied based on num_items

    num_itmes: int
        If item_ids is None will supply num_items IDs

    use_cuda: bool
        Whether to allocate tensors to GPU

    Returns
    -------

    user_var: tensor
        A tensor of user_ids of size (num_items,)

    item_var: tensor
        A tensor of item_ids of size (num_items,)

    """
    if item_ids is None:
        item_ids = np.arange(num_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if item_ids.size()[0] != user_ids.size(0):
        user_ids = user_ids.expand(item_ids.size())

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()


def shuffle(arrays, random_state=None):
    """
    Shuffle all arrays in a list, preserving relative ordering

    Parameters
    ----------

    arrays: list
        A list of arrays, with the same length

    random_state: Numpy Random State Object

    Returns
    ----------

    A tuple of shuffled arrays

    """
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)