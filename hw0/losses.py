"""
Loss functions for recommender models.
"""

import torch
import utils

def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    loss = (1.0 - torch.sigmoid(positive_predictions -
                                negative_predictions))

    if mask is None:
        mask = torch.ones_like(loss)

    mask = mask.float()
    loss = loss * mask
    return loss.sum() / mask.sum()


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    utils.assert_no_grad(observed_ratings)
    return ((observed_ratings - predicted_ratings) ** 2).mean()

