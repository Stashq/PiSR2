import pandas as pd
import numpy as np

from src.models.recommender import Recommender
from src.util import metrics


def test_mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender
):
    mean, ranks = metrics.mean_reciprocal_rank(test_discretized_ratings, model)

    assert isinstance(mean, float)
    assert mean == 0.5

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == [0.5, 1.0, 0.0]


def test_mean_average_precision(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender,
    N: int
):
    mean, ranks = metrics.mean_average_precision(test_discretized_ratings, model, N)
    trues = [(1/2 + 2/5) / 2, 1, 0]
    assert isinstance(mean, float)
    assert mean == np.mean(trues)

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == trues
