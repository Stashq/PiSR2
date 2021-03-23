import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from src.util.discretizer import RatingDiscretizer
from src.util.data import get_train_test_ratings
import pickle
import torch

column_list = [
    "id",
    "adult",
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "original_language",
    "belongs_to_collection",
    "spoken_languages",
    "genres",
]

EMBEDDING_PATH = Path("../data/emb.pkl")
MOVIES_PATH = Path("../data/movies_metadata.csv")
RATINGS_PATH = Path("../data/ratings_small.csv")


def get_dataset_eval(Embeddings=False) -> pd.DataFrame:
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    to_drop = ["1997-08-20", "2012-09-29", "2014-01-01"]
    for drop_error in to_drop:
        movies = movies[movies.id != drop_error]
    movies.id = movies.id.astype("int64")
    dataset = ratings.merge(movies[column_list], left_on="movieId", right_on="id")

    dataset = dataset.dropna()

    budget_scaler = preprocessing.StandardScaler().fit(
        dataset.budget.to_numpy().reshape(-1, 1)
    )

    lang_encoder = preprocessing.LabelEncoder().fit(dataset.original_language)

    genres_encoder = preprocessing.LabelEncoder().fit(dataset.genres)

    spoken_languages_encoder = preprocessing.LabelEncoder().fit(
        dataset.spoken_languages
    )

    adult_encoder = preprocessing.LabelEncoder().fit(dataset.adult)

    belongs_to_collection_encoder = preprocessing.LabelEncoder().fit(
        dataset.belongs_to_collection
    )

    dataset.belongs_to_collection = belongs_to_collection_encoder.transform(
        dataset.belongs_to_collection
    )
    dataset.original_language = lang_encoder.transform(dataset.original_language)
    dataset.adult = adult_encoder.transform(dataset.adult)
    dataset.spoken_languages = spoken_languages_encoder.transform(
        dataset.spoken_languages
    )
    dataset.genres = genres_encoder.transform(dataset.genres)
    dataset.budget = budget_scaler.transform(dataset.budget.to_numpy().reshape(-1, 1))
    dataset.popularity = dataset.popularity.astype("float64")

    dataset.userId = dataset.userId.astype(int)
    dataset.movieId = dataset.movieId.astype(int)

    if Embeddings:
        movies_emb = pickle.load(open(EMBEDDING_PATH, "rb"))
        dataset = dataset.merge(
            movies_emb[["id", "vector"]], left_on="movieId", right_on="id"
        )
        dataset = dataset.drop(["id_x", "id_y"], axis=1)
    else:
        dataset = dataset.drop(["id"], axis=1)
    dataset = dataset.drop(["timestamp"], axis=1)

    return dataset


def get_dataset():
    dataset = get_dataset_eval(True)
    train_ratings, test_ratings = get_train_test_ratings(dataset)

    target_train = torch.Tensor(train_ratings.rating.values)
    target_test = torch.Tensor(test_ratings.rating.values)
    train_ratings = train_ratings.drop(["movieId", "rating"], axis=1)
    test_ratings = test_ratings.drop(["movieId", "rating"], axis=1)

    train_ratings_t = torch.Tensor(train_ratings.drop(["vector"], axis=1).values.astype(float))
    test_ratings_t = torch.Tensor(test_ratings.drop(["vector"], axis=1).values.astype(float))

    vec_train = torch.Tensor(list(train_ratings.vector.values))
    vec_test = torch.Tensor(list(test_ratings.vector.values))

    train = torch.cat((vec_train, train_ratings_t), dim=1)
    test = torch.cat((vec_test, test_ratings_t), dim=1)

    dataset_tensor_train = TensorDataset(train, target_train)
    dataset_tensor_test = TensorDataset(test, target_test)

    return dataset_tensor_train, dataset_tensor_test
