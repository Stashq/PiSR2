from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import MSELoss

from src.models.rnn.data import get_dataset
from src.models.clustering.model import KMeansClustering
from src.models.rnn.trainer import Trainer
from src.util import metrics
from src.util.data import get_interactions, get_sparsity_factor, get_train_test_ratings
from src.util.discretizer import RatingDiscretizer

pio.renderers.default = "notebook"

RATINGS_PATH = Path("data/ratings_small.csv")



ratings = pd.read_csv(RATINGS_PATH)

user_encoder = LabelEncoder()
user_encoder.fit(ratings["userId"].values)

movie_encoder = LabelEncoder()
movie_encoder.fit(ratings["movieId"].values)



ratings["rating"] /= ratings["rating"].values.max()

train_ratings, test_ratings = get_train_test_ratings(ratings)


train_interactions = get_interactions(train_ratings, user_encoder, movie_encoder)
test_interactions = get_interactions(test_ratings, user_encoder, movie_encoder)

train_sparsity = get_sparsity_factor(train_interactions)
test_sparsity = get_sparsity_factor(test_interactions)



print(f"Train sparsity: {(train_sparsity * 100):.3f}%")
print(f"Test sparsity: {(test_sparsity * 100):.3f}%")



# ? binarization is used only to validate ranking metrics

rating_discretizer = RatingDiscretizer()
train_discretized_ratings = rating_discretizer.fit_transform(train_ratings)
test_discretized_ratings = rating_discretizer.transform(test_ratings)



train_ratings["userId"] = user_encoder.transform(train_ratings["userId"].values)
test_ratings["userId"] = user_encoder.transform(test_ratings["userId"].values)

train_ratings["movieId"] = movie_encoder.transform(train_ratings["movieId"].values)
test_ratings["movieId"] = movie_encoder.transform(test_ratings["movieId"].values)



model = KMeansClustering(
    user_encoder=user_encoder,
    movie_encoder=movie_encoder,
    n_clusters=96
)

model.fit(train_interactions)



mean_reciprocal_rank, reciprocal_ranks = metrics.mean_reciprocal_rank(
    test_discretized_ratings,
    model
)
mean_average_rank, average_ranks  = metrics.mean_average_precision(
    test_discretized_ratings,
    model,
    N=10
)
mean_ndcg, ndcg_ranks = metrics.mean_ndcg(
    test_discretized_ratings,
    model
)
print(f"Mean Reciprocal Rank: {(mean_reciprocal_rank * 100):.2f}%")
print(f"Mean Average Rank: {(mean_average_rank * 100):.2f}%")
print(f"Mean NDCG: {(mean_ndcg * 100):.2f}%")



fig = px.histogram(
    x=reciprocal_ranks,
    marginal="box",
    title="Reciprocal Rank Distribution",
    labels={
        "x": "Reciprocal Rank"
    },
)

fig.show()



fig = px.histogram(
    x=average_ranks,
    marginal="box",
    title="Average Rank Distribution",
    labels={
        "x": "Average Rank"
    },
)

fig.show()



# fig = px.histogram(
#     x=ndcg_ranks,
#     marginal="box",
#     title="Reciprocal Rank Distribution",
#     labels={
#         "x": "Reciprocal Rank"
#     },
# )

# fig.show()


