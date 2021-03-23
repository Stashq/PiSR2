import pandas as pd
import torch
from torch.utils.data import TensorDataset


def get_dataset(ratings: pd.DataFrame, device: str) -> TensorDataset:

    users = ratings["userId"].values
    movies = ratings["movieId"].values
    ratings = ratings["rating"].values

    users = torch.LongTensor(users).to(device)
    movies = torch.LongTensor(movies).to(device)
    ratings = torch.FloatTensor(ratings).to(device)

    return TensorDataset(users, movies, ratings)
