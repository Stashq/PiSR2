from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from src.models.lstm.model import LSTMRatingsModel

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class Trainer:
    def __init__(
        self,
        loss: _Loss,
        regularizers: List[_Loss],
        lr: float,
        weight_decay: float = 0,
        epochs: int = 50,
        batch_size: int = 1_000,
        verbose: int = 0,
    ):
        super(Trainer, self).__init__()

        self.loss = loss
        self.regularizers = regularizers

        self.LR = lr
        self.WEIGHT_DECAY = weight_decay
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.VERBOSE = verbose

        self.train_loss_history: List[float]
        self.test_loss_history: List[float]

    def fit(
        self,
        model: LSTMRatingsModel,
        train_dataset: TensorDataset,
        test_dataset: TensorDataset,
    ):

        self.train_loss_history = []
        self.test_loss_history = []

        optimizer = optim.Adam(
            model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY
        )

        test_users, test_movies, test_ratings = test_dataset.tensors

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
        )

        model.to(DEVICE)

        for epoch in tqdm(range(0, self.EPOCHS), desc="Training"):
            train_loss = 0

            for users_batch, movies_batch, ratings_batch in data_loader:
                optimizer.zero_grad()

                prediction = model(users_batch, movies_batch)
                loss = self.loss(prediction, ratings_batch)

                for regularizer in self.regularizers:
                    loss += regularizer(prediction)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(data_loader)

            model.eval()
            with torch.no_grad():

                test_prediction = model(test_users, test_movies)
                test_loss = self.loss(test_prediction, test_ratings).item()

                for regularizer in self.regularizers:
                    test_loss += regularizer(test_prediction).item()

            model.train()

            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            if self.VERBOSE:
                msg = f"Train loss: {train_loss:.3f}, "
                msg += f"Test loss: {test_loss:.3f}"
                print(msg)

    def get_loss_history(self) -> pd.DataFrame:
        loss_history = {
            "epoch": list(range(0, self.EPOCHS)) + list(range(0, self.EPOCHS)),
            "value": self.train_loss_history + self.test_loss_history,
            "loss": ["train"] * self.EPOCHS + ["test"] * self.EPOCHS,
        }

        loss_history = pd.DataFrame(loss_history, columns=loss_history.keys())
        return loss_history
