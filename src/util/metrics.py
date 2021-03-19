from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score

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


def mean_ndcg(
        test_discretized_ratings: pd.DataFrame,
        model: Recommender
) -> Tuple[float, List[float]]:
    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_movies_score = model.predict_scores(user_id)

        pred_movies = pred_movies_score[0]
        pred_score = pred_movies_score[1]

        y_true = np.asarray([user_ratings['liked'].values])
        y_idx = user_ratings['movieId'].values

        indices = [pred_movies.index(movie_id) for movie_id in y_idx]

        pred_movies = np.take(pred_movies, indices)
        pred_score = np.asarray([np.take(pred_score, indices)])
        if pred_score.size > 1:  # If there is no enough data -then ignore score counting for that user
            ndcg = ndcg_score(y_true, pred_score)
            ranks.append(ndcg)

    return np.mean(ranks), ranks
