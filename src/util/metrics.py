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

        test_movies = set(user_ratings["movieId"].values)
        test_liked_movies = set(user_ratings[user_ratings["liked"]]["movieId"].values)

        pred_movies = [
            pred_movie in test_liked_movies
            for pred_movie in pred_movies
            if pred_movie in test_movies
        ]

        indices = np.nonzero(pred_movies)[0]
        min_index = indices.min() if indices.size else float("inf")

        rank = 1 / (min_index + 1)
        ranks.append(rank)

    return np.mean(ranks), ranks


def mean_average_precision(
    test_discretized_ratings: pd.DataFrame, model: Recommender, N: int
) -> Tuple[float, List[float]]:
    """
    Calculate mean average precision.

    Parameters
    ----------
    test_discretized_ratings : pd.DataFrame
        Dataframe containing true users films ratings.
    model : Recommender
        Tested model.
    N : int
        Number of movies taken into account in recommendation for single user.

    Returns
    -------
    Tuple[float, List[float]]
        1. Mean average precision averaged for all users.
        2. List of average precision for all users separately.
    """
    ranks = []
    test_discretized_ratings = test_discretized_ratings.groupby("userId")
    iterator = tqdm(test_discretized_ratings, desc="Testing predictions")
    for user_id, user_ratings in iterator:
        pred_movies = model.predict(user_id)[:N]
        N_user_favourites = user_ratings[user_ratings["liked"]]
        N_user_favourites = N_user_favourites.sort_values(by="rating", ascending=False)
        N_user_favourites = N_user_favourites["movieId"].values

        user_score = get_user_AP_score(pred_movies, N_user_favourites, N)
        ranks.append(user_score)

    return np.mean(ranks), ranks


def get_user_AP_score(
    pred_movies: List[int], N_user_favourites: List[int], N: int
) -> float:
    """Calculate average precision for user.

    Parameters
    ----------
    pred_movies : List[int]
        Sorted (descending) list of movies predicted by model for user to recommend.
    N_user_favourites : List[int]
        Most favourite user N movies.
    N : int
        Number of movies taken into account in recommendation for single user.

    Returns
    -------
    float
        Average precision for user recommendation.
    """
    if N > len(pred_movies):
        N = len(pred_movies)
    sublist_scores = []
    for sublist_len in range(1, N + 1):
        n_correct = 0
        for i, movie_id in enumerate(reversed(pred_movies[:sublist_len])):
            if i == 0 and movie_id not in N_user_favourites:
                break
            if movie_id in N_user_favourites:
                n_correct += 1
        if n_correct != 0:
            sublist_scores.append(n_correct / sublist_len)
    if len(sublist_scores) == 0:
        return 0.0
    else:
        return np.mean(sublist_scores)


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


def coverage(
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
    prediction_coverage = round(unique_predictions / (len(all_movies) * 1.0), 2)
    return prediction_coverage


def rmse(
    test_ratings: pd.DataFrame,
    model: Recommender,
) -> float:

    predicted = []

    iterator = tqdm(
        test_ratings.iterrows(), total=len(test_ratings), desc="Testing predictions"
    )
    for index, row in iterator:
        pred_movies = model.predict_score(row.userId, row.movieId)
        predicted.append(pred_movies)

    e = np.array(predicted) - test_ratings.rating.values
    se = e ** 2
    rmse = se.mean() ** 0.5
    return rmse
