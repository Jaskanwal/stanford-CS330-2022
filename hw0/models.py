"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.u_factorization = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
        self.q_factorization = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)
        self.a_factorization = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1, sparse=sparse)
        self.b_factorization = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1, sparse=sparse)

        if embedding_sharing:
            self.u_regression = self.u_factorization
            self.q_regression = self.q_factorization
        else:
            self.u_regression = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
            self.q_regression = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)

        # Define hidden layers for regression
        self.layers_regression = []
        for ind in range(len(layer_sizes)-1):
            self.layers_regression.append(nn.Linear(layer_sizes[ind], layer_sizes[ind + 1]))
            self.layers_regression.append(nn.ReLU())
        self.layers_regression.append(nn.Linear(layer_sizes[-1], 1))

        self.layers_regression = nn.Sequential(*self.layers_regression)
        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        # Predictions from the matrix factorization model
        predictions = (
                          (self.u_factorization(user_ids) * self.q_factorization(item_ids)).sum(dim=1) +
                          self.a_factorization(user_ids).squeeze() + self.b_factorization(item_ids).squeeze()
        ).squeeze()

        # Scores from the regression model
        score = self.layers_regression(torch.concat([self.u_regression(user_ids),
                                                     self.q_regression(item_ids),
                                                     self.u_regression(user_ids) * self.q_regression(item_ids)],
                                                    dim=1)
                                       ).squeeze()
        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score