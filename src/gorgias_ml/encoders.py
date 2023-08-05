import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import gorgias_ml.constants as cst


class TicketMessageAverageEncoder(BaseEstimator, TransformerMixin):
    """
    Encode the message as the average of sentence embeddings.
    It adds a new column features.
    """

    def __init__(self, col_name=cst.EMBEDDINGS_COL_NAME, features_name=cst.FEATURES):
        self.col_name = col_name
        self.features_name = features_name

    @staticmethod
    def _to_2d_array(emb: dict[str, list[float]]) -> np.array:
        """
        transforms the dict into a 2D array
        :param emb: dict of sentence embeddings
        :return: 2D array
        """
        return np.array(list(emb.values()))

    def fit(self, x: pd.DataFrame = None, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        x_copy = x.copy()
        x_copy[self.features_name] = x_copy[self.col_name].swifter.apply(
            lambda emb: np.mean(self._to_2d_array(emb), axis=0)
        )

        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)
