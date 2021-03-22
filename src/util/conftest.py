from typing import List, Tuple

import numpy as np
import pandas as pd
from pytest import fixture

from src.models.recommender import Recommender


class DummyRecommender(Recommender):
    def __init__(self):
        super(DummyRecommender, self).__init__()

        self.RECOMMENDATIONS = {
            1: {2: 5.0, 32: 4.5, 4121: 4.2, 5422: 4.1, 12: 3.5, 42: 2.5},
            2: {42: 4.7, 5534: 4.2, 11: 3.0},
            3: {31: 2.75},
        }

    def predict(self, user_id: int) -> List[int]:
        ranking, scores = self.predict_scores(user_id)
        return ranking

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return self.RECOMMENDATIONS[user_id][movie_id]

    def predict_scores(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        movies = np.array(list(self.RECOMMENDATIONS[user_id].keys()))
        scores = np.array(list(self.RECOMMENDATIONS[user_id].values()))
        return movies, scores


@fixture(scope="session")
def model() -> Recommender:
    return DummyRecommender()


@fixture(scope="session")
def test_ratings() -> pd.DataFrame:
    ratings = {
        "userId": [1, 1, 1, 2, 2, 3],
        "movieId": [32, 12, 42, 42, 11, 31],
        "rating": [5.0, 4.0, 2.5, 5.0, 3.5, 2.5],
        "timestamp": [
            1260759144,
            1260759179,
            1260759182,
            1260759185,
            1260759205,
            835355681,
        ],
    }

    return pd.DataFrame(ratings)


@fixture(scope="session")
def test_discretized_ratings(test_ratings: pd.DataFrame) -> pd.DataFrame:
    test_discretized_ratings = test_ratings.copy()
    test_discretized_ratings["liked"] = [True, True, False, True, False, False]

    return test_discretized_ratings


@fixture(scope="session")
def N() -> int:
    return 10
