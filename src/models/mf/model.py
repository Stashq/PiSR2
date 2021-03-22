from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from src.models.recommender import Recommender


class MatrixFactorization(nn.Module, Recommender):
    def __init__(self, users_dim, movies_dim, h_dim=5, use_mlp=False):
        super().__init__()
        self.user_encoder = None
        self.movie_encoder = None

        self.user_embedding = nn.Embedding(users_dim, h_dim)
        self.movie_embedding = nn.Embedding(movies_dim, h_dim)

        self.global_bias = torch.tensor(1.0)
        self.user_bias = nn.Embedding(movies_dim, 1)
        self.movie_bias = nn.Embedding(movies_dim, 1)

        self.use_mlp = use_mlp
        if use_mlp:
            self.microNN = nn.Sequential(
                nn.Linear(2 * h_dim, h_dim),
                nn.Linear(h_dim, 1),
            )

    def initialize(self, pandas_dataframe_reindexed, embedding_rescaler=1.0):
        mean_rating = pandas_dataframe_reindexed.rating.mean()
        self.global_bias = torch.tensor(mean_rating)
        self.user_bias.weight.data = torch.full(self.user_bias.weight.shape, 0.0)
        self.movie_bias.weight.data = torch.full(self.movie_bias.weight.shape, 0.0)

        movie_stats = pandas_dataframe_reindexed.groupby("movieId").agg(
            {"rating": ["mean"]}
        )
        movie_stats.columns = movie_stats.columns.get_level_values(1)

        user_stats = pandas_dataframe_reindexed.groupby("userId").agg(
            {"rating": ["mean"]}
        )
        user_stats.columns = user_stats.columns.get_level_values(1)

        for user_id, user_mean in zip(user_stats["mean"].index, user_stats["mean"]):
            self.user_bias.weight.data[user_id, :] = user_mean - mean_rating

        for movie_id, movie_mean in zip(movie_stats["mean"].index, movie_stats["mean"]):
            self.movie_bias.weight.data[movie_id, :] = movie_mean - mean_rating
        self.user_bias.weight.requires_grad = False
        self.movie_bias.weight.requires_grad = False

        self.user_embedding.weight.data *= embedding_rescaler
        self.movie_embedding.weight.data *= embedding_rescaler

        self.pandas_dataframe_reindexed = pandas_dataframe_reindexed.copy()

    def forward(self, users, movies):
        u, m = self.user_embedding(users), self.movie_embedding(movies)
        rating = u * m
        rating = rating.sum(1, keepdim=True)
        rating += self.global_bias
        rating += self.user_bias(users)
        rating += self.movie_bias(movies)
        if self.use_mlp:
            rating += self.microNN(torch.cat([u, m], dim=1))
        return rating.squeeze()

    def set_label_encoders(
        self, user_encoder: LabelEncoder, movie_encoder: LabelEncoder
    ):
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder

    def assert_encoders_exist(self):
        assert (self.user_encoder is not None) and (
            self.user_encoder is not None
        ), "label encoders not initialized, run .set_label_encoders()"

    def predict(self, user_id: int) -> List[int]:
        """
        Predicts ranking of movies to watch for a user.
        Parameters
        ----------
        user_id : int
            User's id from the data set.
        Returns
        -------
        List[int]
            List of movies ids. Best recommendations first.
        """
        movies, rating = self.predict_scores(user_id)
        return list(movies)

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """
        Predicts score for a given movie that a user would give.
        Parameters
        ----------
        user_id : int
            User's id from the data set.
        movie_id : int
            Movie's id from the data set.
        Returns
        -------
        float
            Predicted movie's score in range [0, 5].
        """
        self.assert_encoders_exist()
        user_id = self.user_encoder.transform([user_id])[0]
        movie_id = self.movie_encoder.transform([movie_id])[0]

        user_id = torch.LongTensor([user_id])
        movie_id = torch.LongTensor([movie_id])

        rating = self.forward(user_id, movie_id)
        rating = rating.cpu().item()

        return rating

    def predict_scores(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts scores for all the movies, that a user would give.
        Parameters
        ----------
        user_id : int
            User's id from the data set.
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]:
            Ranked movies with their scores.
        """
        self.assert_encoders_exist()
        user_id = self.user_encoder.transform([user_id])[0]

        movies = set(range(len(self.movie_encoder.classes_)))
        movies_seen = self.pandas_dataframe_reindexed[
            self.pandas_dataframe_reindexed.userId == 1
        ].movieId

        movies -= set(movies_seen)
        movies = list(movies)
        movies = torch.LongTensor(movies)

        user = torch.LongTensor([user_id] * len(movies))

        ratings = self.forward(user, movies)

        ratings = list(ratings.detach().cpu().numpy())
        movies = list(movies.detach().cpu().numpy())

        ranking = pd.DataFrame(zip(movies, ratings), columns=["movie", "rating"])

        ranking = ranking.sort_values(by="rating", ascending=False)

        movies = ranking["movie"].values
        ratings = ranking["rating"].values

        movies = self.movie_encoder.inverse_transform(movies)

        return movies, ratings
