import numpy as np
import pandas as pd

from src.models.recommender import Recommender
from src.util import metrics
from src.util.metrics import coverage


def test_mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame, model: Recommender
):
    mean, ranks = metrics.mean_reciprocal_rank(test_discretized_ratings, model)

    assert isinstance(mean, float)
    assert mean == 2 / 3

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == [1.0, 1.0, 0.0]


def test_mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender, N: int
):
    mean, ranks = metrics.mean_average_precision(test_discretized_ratings, model, N)
    trues = [(1 / 2 + 2 / 5) / 2, 1, 0]
    assert isinstance(mean, float)
    assert mean == np.mean(trues)

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == trues


def test_mean_ndcg(test_discretized_ratings: pd.DataFrame, model: Recommender):
    mean, ranks = metrics.mean_ndcg(test_discretized_ratings, model)
    assert isinstance(mean, float)
    assert isinstance(ranks, list)
    assert ranks  # check if ranks is not empty list
    assert all(isinstance(rank, float) for rank in ranks)


def test_coverage(test_discretized_ratings: pd.DataFrame, model: Recommender):
    cov = coverage(test_discretized_ratings, model)
    assert isinstance(cov, float)
