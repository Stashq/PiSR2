import pandas as pd
import torch
from torch.utils.data import TensorDataset

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def get_dataset(ratings: pd.DataFrame) -> TensorDataset:

    users = ratings["userId"].values
    movies = ratings["movieId"].values
    ratings = ratings["rating"].values

    users = torch.LongTensor(users).to(DEVICE)
    movies = torch.LongTensor(movies).to(DEVICE)
    ratings = torch.FloatTensor(ratings).to(DEVICE)

    return TensorDataset(users, movies, ratings)
