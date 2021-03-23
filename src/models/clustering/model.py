import pickle
from typing import List, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from src.models.recommender import Recommender


class KMeansClustering(Recommender):
    def __init__(
        self,
        user_encoder: LabelEncoder,
        movie_encoder: LabelEncoder,
        n_clusters: int = 100,
    ):
        super(KMeansClustering, self).__init__()

        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder

        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, interactions: np.ndarray):
        """Learns model on given interactions.
        Saves informations such as:
        - which user belongs to which cluster
        - averaged ratings for cluster members called "club taste"

        Parameters
        ----------
        interactions : np.ndarray
            Interactions matrix. Rows are users, columns are movies.
            Specific cell denotes the rating, how user scored the movie.
            Interactions are encoded to handle continuity of indices.
        """
        self.interactions = interactions
        n_movies = interactions.shape[1]

        self.model.fit(interactions)
        self.labels = self.model.labels_

        self.club_taste = np.zeros(shape=(len(set(self.labels)), n_movies))
        for cluster in set(self.labels):
            users_ids = cluster == self.labels  # filtering users in cluster
            for i in range(n_movies):
                self.club_taste[cluster, i] = self._cluster_avg_movie_rating(
                    users_ids, i
                )

    def _cluster_avg_movie_rating(
        self, users_ids: Union[List[int], List[bool]], movie_id: int
    ) -> float:
        """Calculate rating of movie for cluster.
        It is done by averaging only ratings of users belonging to one cluster.

        Parameters
        ----------
        users_ids : Union[List[int], List[bool]]
            Users belonging to one cluster.
        movie_id : int
            Encoded id of movie.

        Returns
        -------
        float
            Averaged rating of movie.
            If no user has rated movie, returns 0.
        """
        movie_ratings = [
            rate for rate in self.interactions[users_ids, movie_id] if rate > 0.0
        ]
        movie_mean = np.mean(movie_ratings)
        if np.isnan(movie_mean):
            movie_mean = 0.0
        return movie_mean

    def predict(self, user_id: int) -> List[int]:
        user_id = self.user_encoder.transform([user_id])[0]
        cluster = self.labels[user_id]
        ratings = self.club_taste[cluster]

        unrated = np.where(self.interactions[user_id, :] == 0)[0]
        recom_ids = np.argsort(ratings[unrated])[::-1]

        return recom_ids

    def predict_score(self, user_id: int, movie_id: int) -> float:
        user_id = self.user_encoder.transform([user_id])[0]
        cluster = self.labels[user_id]
        return self.club_taste[cluster, movie_id]

    def predict_scores(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        user_id = self.user_encoder.transform([user_id])[0]
        cluster = self.labels[user_id]
        ratings = self.club_taste[cluster]

        unrated = np.where(self.interactions[user_id, :] == 0)[0]
        recom_ids = np.argsort(ratings[unrated])[::-1]
        recom_rates = np.array(sorted(ratings[unrated]))[::-1]

        recom_ids = self.movie_encoder.inverse_transform(recom_ids)

        return recom_ids, recom_rates
