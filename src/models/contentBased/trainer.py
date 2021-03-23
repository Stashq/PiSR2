from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.auto import tqdm

from src.models.contentBased.data import get_dataset

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
        model: nn.Module,
    ):

        self.train_loss_history = []
        self.test_loss_history = []

        optimizer = optim.Adam(
            model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY
        )

        train_dataset, test_dataset = get_dataset()

        data_loader_train = DataLoader(train_dataset, batch_size=self.BATCH_SIZE)

        test_set, test_targets = test_dataset.tensors
        test_set = test_set.to(DEVICE)
        test_targets = test_targets.to(DEVICE)
        model.to(DEVICE)

        for epoch in tqdm(range(0, self.EPOCHS), desc="Training"):
            train_loss = 0

            for train, target in data_loader_train:
                optimizer.zero_grad()
                x = train.to(DEVICE)
                y = target.to(DEVICE).reshape((-1,1))
                prediction = model(x)
                loss = self.loss(prediction, y)

                for regularizer in self.regularizers:
                    loss += regularizer(prediction)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            test_prediction = model(test_set)
            test_loss = self.loss(test_prediction, test_targets.reshape((-1,1))).item()

            for regularizer in self.regularizers:
                test_loss += regularizer(test_prediction).item()

            train_loss /= len(data_loader_train)

            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            if self.VERBOSE:
                msg = f"Train loss: {train_loss:.3f}, "
                msg += f"Test loss: {test_loss:.3f}"
                tqdm.write(msg)

    def get_loss_history(self) -> pd.DataFrame:
        loss_history = {
            "epoch": list(range(0, self.EPOCHS)) + list(range(0, self.EPOCHS)),
            "value": self.train_loss_history + self.test_loss_history,
            "loss": ["train"] * self.EPOCHS + ["test"] * self.EPOCHS,
        }

        loss_history = pd.DataFrame(loss_history, columns=loss_history.keys())
        return loss_history
