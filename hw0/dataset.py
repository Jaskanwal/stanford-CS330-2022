"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os
import requests
import h5py
import scipy.sparse as sp
import numpy as np

URL_PREFIX = 'https://github.com/maciejkula/recommender_datasets/releases/download'

VERSION = 'v0.2.0'

VARIANTS = ('100K', '1M', '10M', '20M')

DATA_DIR = 'movielens_data'


def index_or_none(array, shuffle_index):
    if array is None:
        return None
    else:
        return array[shuffle_index]

def download(url, dest_path, data_dir=DATA_DIR):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def get_data(url, dest_subdir, dest_filename, download_if_missing=True):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), dest_subdir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)


    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path):
        if download_if_missing:
            download(url, dest_path)
        else:
            raise IOError('Dataset missing.')

    return dest_path


class Interactions:
    """
    Interactions object. Contains pairs of user-item
    interactions and corresponding ratings.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def _check(self):

        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError('Invalid {} dimensions: length '
                                 'must be equal to number of interactions'
                                 .format(name))

    def shuffle_interactions(self, random_state=None):
        """
        Shuffle interactions.

        Parameters
        ----------

        interactions: class:Interactions
            The interactions to shuffle.
        random_state: np.random.RandomState, optional
            The random state used for the shuffle.

        Returns
        -------

        interactions: class:Interactions
            The shuffled interactions.
        """

        if random_state is None:
            random_state = np.random.RandomState()

        shuffle_indices = np.arange(len(self.user_ids))
        random_state.shuffle(shuffle_indices)

        return Interactions(self.user_ids[shuffle_indices],
                            self.item_ids[shuffle_indices],
                            ratings=index_or_none(self.ratings,
                                                  shuffle_indices),
                            timestamps=index_or_none(self.timestamps,
                                                     shuffle_indices),
                            weights=index_or_none(self.weights,
                                                  shuffle_indices),
                            num_users=self.num_users,
                            num_items=self.num_items)


    def random_train_test_split(self, test_fraction=0.2, random_state=None):
        """
        Randomly split interactions between training and testing.

        Parameters
        ----------

        interactions: :class:Interactions
            The interactions to shuffle.
        test_percentage: float, optional
            The fraction of interactions to place in the test set.
        random_state: np.random.RandomState, optional
            The random state used for the shuffle.

        Returns
        -------

        (train, test): (class:Interactions,
                        class:Interactions)
             A tuple of (train data, test data)
        """

        interactions = self.shuffle_interactions(random_state=random_state)

        cutoff = int((1.0 - test_fraction) * len(interactions))

        train_idx = slice(None, cutoff)
        test_idx = slice(cutoff, None)

        train = Interactions(interactions.user_ids[train_idx],
                             interactions.item_ids[train_idx],
                             ratings=index_or_none(interactions.ratings,
                                                   train_idx),
                             timestamps=index_or_none(interactions.timestamps,
                                                      train_idx),
                             weights=index_or_none(interactions.weights,
                                                   train_idx),
                             num_users=interactions.num_users,
                             num_items=interactions.num_items)
        test = Interactions(interactions.user_ids[test_idx],
                            interactions.item_ids[test_idx],
                            ratings=index_or_none(interactions.ratings,
                                                  test_idx),
                            timestamps=index_or_none(interactions.timestamps,
                                                     test_idx),
                            weights=index_or_none(interactions.weights,
                                                  test_idx),
                            num_users=interactions.num_users,
                            num_items=interactions.num_items)

        return train, test

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()



def _get_movielens(dataset):
    extension = '.hdf5'

    path = get_data('/'.join((URL_PREFIX, VERSION, dataset + extension)),
                    os.path.join('movielens', VERSION),
                    'movielens_{}{}'.format(dataset, extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])


def get_movielens_dataset(variant='100K'):
    """
    Download and return one of the Movielens datasets.

    Parameters
    ----------

    variant: string, optional
         String specifying which of the Movielens datasets
         to download. One of ('100K', '1M', '10M', '20M').

    Returns
    -------

    Interactions: class:Interactions
        instance of the interactions class
    """

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    url = 'movielens_{}'.format(variant)

    return Interactions(*_get_movielens(url))
