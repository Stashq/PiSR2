from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.models.recommender import Recommender


def mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender
) -> Tuple[float, List[float]]:

    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_movies = model.predict(user_id)

        test_liked_user_ratings = user_ratings[user_ratings["liked"]]
        test_liked_user_movies = test_liked_user_ratings["movieId"].values

        indices = [pred_movies.index(movie_id) for movie_id in test_liked_user_movies]
        min_index = min(indices) if indices else float('inf')

        rank = 1 / (min_index + 1)
        ranks.append(rank)

    return np.mean(ranks), ranks


def mean_average_precision(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender
) -> Tuple[float, List[float]]:
    pass
