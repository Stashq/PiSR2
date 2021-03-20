from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class CustomScaler(TransformerMixin, BaseEstimator):
    """Constructs a sklearn based transformer working on pandas dataframes
    transforming user ratings to +1 or -1 if a rating is above or below
    user average. If a user has only a single rating then
    global average rating is used.
    Parameters
    ----------
    squeeze_movie_indexes : bool, default=True
        Indicate if movie indexes should be reindexed
        from 0 to len(ratings.movieId.unique()).
    Example
    --------
    >>> import pandas as pd
    >>> from src.util.discretizer import CustomScaler
    >>> transformer = CustomScaler()
    >>> X = pd.read_csv('ratings_small.csv')
    >>> X_train, X_test = sklearn.model_selection.train_test_split(X)
    >>> transformer.fit(X)
    >>> transformer.transform(X_test)
            userId  movieId  rating   timestamp
    38501      282      195    -1.0  1111494823
    99428      665     3110     1.0   995232733
    76284      529      959    -1.0   960052682
    ...        ...      ...     ...         ...
    """

    def __init__(self, *, squeeze_movie_indexes=True):
        self.squeeze_movie_indexes = squeeze_movie_indexes

    def fit(self, X, y=None):
        """Fit transformer by checking X.
        If ``validate`` is ``True``, ``X`` will be checked.
        Parameters
        ----------
        X : pandas dataframe containing columns ['userId', 'movieId', 'rating']
        Returns
        -------
        self
        """
        # X = self._check_input(X)  # in future version

        user_stats = X.groupby("userId").agg({"rating": ["mean", "count"]})
        user_stats.columns = user_stats.columns.get_level_values(1)

        self.means = user_stats["mean"]

        # for users with a single rating just assume global mean
        global_mean = X.rating.mean()
        self.means.update(
            pd.Series(global_mean, self.means[user_stats["count"] == 1].index)
        )

        if self.squeeze_movie_indexes:
            self.old_movie_ids = X.movieId.unique()
            self.new_movie_ids = pd.Series(
                {i: new for new, i in enumerate(self.old_movie_ids)}
            )
        return self

    def transform(self, X):
        """Transform X using the forward function.
        Parameters
        ----------
        X : pandas dataframe containing columns ['userId', 'movieId', 'rating']
        Returns
        -------
        X_out : pandas dataframe containing columns ['userId','rating']
            where rating equals 1 or -1 if user likes the movie or not
            or NaN if user was not present while fitting the CustomScaler
            if more columns were passed to the transform function,
            only 'rating' column is modified and the rest is returned unchanged
        """
        Xcopy = X.copy()
        valid_user_ids = set(X["userId"]) & set(self.means.index)
        ratings = X.set_index("userId").rating
        ratings = np.sign(ratings - self.means[valid_user_ids]).values
        Xcopy.rating.update(pd.Series(ratings, X.rating.index))

        if self.squeeze_movie_indexes:
            bad_movie_ids = set(X["movieId"]) - set(self.new_movie_ids.index)
            assert len(bad_movie_ids) == 0, (
                "passed movieId not present during training\n"
                "consider using CustomScaler(squeeze_movie_indexes=False)\n"
                "to ignore reindexing or clean up the data to be transform"
            )
            Xcopy.movieId.update(
                pd.Series(self.new_movie_ids[X.movieId].values, X.movieId.index)
            )

        return Xcopy

    def inverse_transform(self, X):
        """Transform X using the inverse function.
        Parameters
        ----------
        X : pandas dataframe with columns ['userId', 'movieId', 'rating']
        Returns
        -------
        X_out : pandas dataframe with columns ['userId', 'movieId', 'rating']
            with mapped back column 'movieId'.
            Notice that there is no reverse mapping of ratings
            as this information is permanently lost by the transformation.
        """
        X.movieId.update(pd.Series(self.old_movie_ids[X.movieId], X.movieId.index))
        return X
