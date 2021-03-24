import pickle
from abc import ABC
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.content_based.data import get_dataset_eval
from src.models.recommender import Recommender

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class ContentBaseRecommenderSystem(nn.Module, Recommender):
    def __init__(self, input_size, hidden_feature_size):

        nn.Module.__init__(self)
        Recommender.__init__(self)
        self.movies = get_dataset_eval(embeddings=True)

        self.fc1 = nn.Linear(input_size, hidden_feature_size)
        self.fc2 = nn.Linear(hidden_feature_size, hidden_feature_size)
        self.fc3 = nn.Linear(hidden_feature_size, hidden_feature_size)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.out_layer = nn.Linear(hidden_feature_size, 1)
        self.act_fn = F.relu
        self.setup()

    def setup(self):
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.act_fn(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.act_fn(x)

        x = self.act_fn(self.fc3(x))
        out = torch.sigmoid(self.out_layer(x))

        return out * self.MAX_RATING

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
        df = self.movies.copy()

        df = df.loc[df["movieId"] == movie_id]
        df_features = df.drop(["movieId", "rating", "vector"], axis=1).loc[0]
        df_vector = df.vector.loc[0]
        x_input = torch.Tensor(df_features.values)
        vec_input = torch.Tensor(list(df_vector))
        input = torch.cat((x_input, vec_input), dim=0).to(DEVICE)
        with torch.no_grad():
            prediction = self.forward(input).item()
            prediction = prediction.to("cpu").detach().numpy()
        return prediction

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
        df = self.movies.copy()
       # print('inside normal df', len(df.index))
        #df = df[df.userId != user_id]
        df.userId = user_id
        #print('inside userId df', len(df.index))
        df = df.drop_duplicates("movieId")
      #  print('after duplicates drop df', len(df.index))
        movieIds = df.movieId.values
        df_features = df.drop(["movieId", "rating", "vector"], axis=1)


        df_vector = df.vector.values
      #  print('inside' ,len(movieIds))
        x_input = torch.Tensor(df_features.values)
        vec_input = torch.Tensor(list(df_vector))
        input = torch.cat((x_input, vec_input), dim=1).to(DEVICE)
        with torch.no_grad():
            prediction = self.forward(input)
            prediction = prediction.to("cpu").detach().numpy()

        ranking = pd.DataFrame(zip(movieIds, prediction), columns=["movie", "rating"])

        ranking = ranking.sort_values(by="rating", ascending=False)
        return ranking.movie.values, np.concatenate(ranking.rating.values, axis=0)

    def get_user_data(self, user_id: int, exlude=False) -> pd.DataFrame:
        df = self.movies.copy()
        if exlude:
            df = df.loc[df["userId"] != user_id]
        else:
            df = df.loc[df["userId"] == user_id]
        return df
