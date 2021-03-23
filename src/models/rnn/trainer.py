from typing import List

import pandas as pd
import torch
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.models.rnn.model import RNNRatings


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
        model: RNNRatings,
        train_dataset: TensorDataset,
        test_dataset: TensorDataset,
    ):

        self.train_loss_history = []
        self.test_loss_history = []

        optimizer = optim.Adam(
            model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY
        )

        test_users, test_movies, test_ratings = test_dataset.tensors

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
        )

        test_data_loader = DataLoader(
            test_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

        for epoch in tqdm(range(0, self.EPOCHS), desc="Training"):
            train_loss = 0

            for users_batch, movies_batch, ratings_batch in train_data_loader:
                optimizer.zero_grad()

                loss = self.step_loss(
                    model,
                    self.loss,
                    self.regularizers,
                    users_batch,
                    movies_batch,
                    ratings_batch,
                )

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_data_loader)
            self.train_loss_history.append(train_loss)

            test_loss = 0

            model.eval()
            with torch.no_grad():

                for users_batch, movies_batch, ratings_batch in test_data_loader:

                    test_loss += self.step_loss(
                        model,
                        self.loss,
                        self.regularizers,
                        users_batch,
                        movies_batch,
                        ratings_batch,
                    ).item()

                test_loss /= len(test_data_loader)

            model.train()

            self.test_loss_history.append(test_loss)

            if self.VERBOSE:
                msg = f"Train loss: {train_loss:.3f}, "
                msg += f"Test loss: {test_loss:.3f}"
                print(msg)

    def step_loss(
        self,
        model: RNNRatings,
        loss: _Loss,
        regularizers: List[_Loss],
        users_batch: torch.Tensor,
        movies_batch: torch.Tensor,
        ratings_batch: torch.Tensor,
    ):
        prediction = model(users_batch, movies_batch)
        loss_ = loss(prediction, ratings_batch)

        for regularizer in regularizers:
            loss_ += regularizer(prediction)

        return loss_

    def get_loss_history(self) -> pd.DataFrame:
        loss_history = {
            "epoch": list(range(0, self.EPOCHS)) + list(range(0, self.EPOCHS)),
            "value": self.train_loss_history + self.test_loss_history,
            "loss": ["train"] * self.EPOCHS + ["test"] * self.EPOCHS,
        }

        loss_history = pd.DataFrame(loss_history, columns=loss_history.keys())
        return loss_history
