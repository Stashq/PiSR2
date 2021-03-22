from src.models.hierarchical import HierarchicalClustering
from src.models.kmeans import KMeansClustering
from src.util.data import get_train_test_ratings, get_interactions, get_sparsity_factor

import pandas as pd
import pickle as plk
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

def save(obj, path):
    with open('src/models/dumps/kmeans.pickle', 'wb') as f:
        plk.dump(obj, f)

def load(path):
    with open('src/models/dumps/kmeans.pickle', 'rb') as f:
        return plk.load(f)

RATINGS_PATH = Path("data/ratings_small.csv")
MODEL_PATH   = Path("src/models/dumps/kmeans.pickle")

ratings = pd.read_csv(RATINGS_PATH)

user_encoder = LabelEncoder()
user_encoder.fit(ratings['userId'].values)

movie_encoder = LabelEncoder()
movie_encoder.fit(ratings['movieId'].values)

train_ratings, test_ratings = get_train_test_ratings(ratings)

train_interactions = get_interactions(
    train_ratings,
    user_encoder,
    movie_encoder
)

test_interactions = get_interactions(
    test_ratings,
    user_encoder,
    movie_encoder
)

train_interactions /= 5
test_interactions /= 5

train_sparsity = get_sparsity_factor(train_interactions)
test_sparsity = get_sparsity_factor(test_interactions)

print(f'Train sparsity: {(train_sparsity * 100):.3f}%')
print(f'Test sparsity: {(test_sparsity * 100):.3f}%')

model = KMeansClustering(user_encoder, movie_encoder, n_clusters=96)
if os.path.isfile(MODEL_PATH):
    model = load(MODEL_PATH)
else:
    model.fit(train_interactions)
    save(model, MODEL_PATH)

print(model.predict_score(ratings.iloc[0, 0], ratings.iloc[0, 1]))
print(ratings.iloc[0])


# # testowanie hierarchicznej klasteryzacji
# model = HierarchicalClustering(distance_threshold=15.0)
# model.fit(train_interactions)
# model.plot_dendrogram()
# labels = model.model.labels_
# score = silhouette_score(train_interactions, labels)
# print(len(set(labels)), score)
# x=1

# # testowanie KMeans
# import csv
# scores_df = pd.read_csv("data/kmeans_scores.csv", index_col="k")
# for i in range(2, 12, 1):
#     if i not in scores_df.index:
#         model = KMeansClustering(i)
#         model.fit(train_interactions)
#         labels = model.model.labels_
#         score = silhouette_score(train_interactions, labels)
#         new = pd.DataFrame(data=[[i, score]], columns=["k", "silhouette"]).set_index('k')
#         scores_df = scores_df.append(new)
#         print(i, score)
# scores_df.to_csv("data/kmeans_scores.csv")
# x=1

