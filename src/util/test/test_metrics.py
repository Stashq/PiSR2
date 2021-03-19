import pandas as pd

from src.models.recommender import Recommender
from src.util import metrics
from src.util.metrics import covrage


def test_mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame, model: Recommender
):
    mean, ranks = metrics.mean_reciprocal_rank(test_discretized_ratings, model)

    assert isinstance(mean, float)
    assert mean == 0.5

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == [0.5, 1.0, 0.0]


def test_mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender
):
    mean, ranks = metrics.mean_average_precision(test_discretized_ratings, model)

    assert isinstance(mean, float)
    assert mean == 0.5  # ! TODO: fix later

    assert isinstance(ranks, list)
    assert all(isinstance(rank, float) for rank in ranks)
    assert ranks == [0.5, 1.0, 0.0]  # ! TODO: fix later


def test_mean_ngdc(test_discretized_ratings: pd.DataFrame, model: Recommender):
    mean, ranks = metrics.mean_ndcg(test_discretized_ratings, model)
    assert isinstance(mean, float)
    assert isinstance(ranks, list)
    assert ranks  # check if ranks is not empty list
    assert all(isinstance(rank, float) for rank in ranks)

    pass


def test_coverage(test_discretized_ratings: pd.DataFrame, model: Recommender):
    cov = covrage(test_discretized_ratings, model)
    assert isinstance(cov, float)
