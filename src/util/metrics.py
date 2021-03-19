from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

from src.models.recommender import Recommender


def mean_reciprocal_rank(
    test_discretized_ratings: pd.DataFrame, model: Recommender
) -> Tuple[float, List[float]]:
    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_movies = model.predict(user_id)

        test_liked_user_ratings = user_ratings[user_ratings["liked"]]
        test_liked_user_movies = test_liked_user_ratings["movieId"].values

        # movie_ids = np.where(np.isin(pred_movies, movie_ids))
        indices = np.argwhere(np.isin(pred_movies, test_liked_user_movies)).flatten()
        min_index = indices.min() if indices.size else float("inf")

        rank = 1 / (min_index + 1)
        ranks.append(rank)

    return np.mean(ranks), ranks


def mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender
) -> Tuple[float, List[float]]:
    pass


def mean_ndcg(
    test_discretized_ratings: pd.DataFrame, model: Recommender
) -> Tuple[float, List[float]]:
    ranks = []

    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")

    for user_id, user_ratings in iterator:
        pred_movies, pred_scores = model.predict_scores(user_id)

        user_liked = user_ratings["liked"].values
        movie_ids = user_ratings["movieId"].values

        movie_ids = np.where(np.isin(pred_movies, movie_ids))
        pred_score = pred_scores[movie_ids]

        if (
            pred_score.size > 1
        ):  # If there is no enough data -then ignore score counting for that user
            ndcg = ndcg_score(user_liked.reshape(1, -1), [pred_score])
            ranks.append(ndcg)

    return np.mean(ranks), ranks


def covrage(
    test_discretized_ratings: pd.DataFrame,
    model: Recommender,
) -> float:

    predicted = []
    all_movies = pd.unique(test_discretized_ratings["movieId"])

    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")
    for user_id, user_ratings in iterator:
        pred_movies = model.predict(user_id)
        predicted.append(pred_movies)

    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_predictions = len(set(predicted_flattened))
    prediction_coverage = round(unique_predictions / (len(all_movies) * 1.0) * 100, 2)
    return prediction_coverage
