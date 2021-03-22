import torch


def get_dataloaders(train_ratings, test_ratings, user_encoder, movie_encoder):
    train_ratings_reindexed = train_ratings.copy()
    train_ratings_reindexed["userId"] = user_encoder.transform(
        train_ratings_reindexed["userId"].values
    )
    train_ratings_reindexed["movieId"] = movie_encoder.transform(
        train_ratings_reindexed["movieId"].values
    )

    test_ratings_reindexed = test_ratings.copy()
    test_ratings_reindexed["userId"] = user_encoder.transform(
        test_ratings_reindexed["userId"].values
    )
    test_ratings_reindexed["movieId"] = movie_encoder.transform(
        test_ratings_reindexed["movieId"].values
    )

    # torch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_ratings_reindexed["userId"].values),
        torch.tensor(train_ratings_reindexed["movieId"].values),
        torch.tensor(train_ratings_reindexed["rating"].values, dtype=torch.float),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_ratings_reindexed["userId"].values),
        torch.tensor(test_ratings_reindexed["movieId"].values),
        torch.tensor(test_ratings_reindexed["rating"].values, dtype=torch.float),
    )

    batch_size = 10000
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
