import ast

import pandas as pd
import swifter
from sklearn.base import BaseEstimator, TransformerMixin

import gorgias_ml.constants as cst


class EmbeddingsCleaner(BaseEstimator, TransformerMixin):
    """Clean embeddings column"""

    def __init__(self, col_name: str = cst.EMBEDDINGS_COL_NAME):
        self.col_name = col_name

    @staticmethod
    def normalize_embeddings(s: str) -> dict[str, list]:
        """
        clean embeddings column
        :param s: string column presenting the dict of sentence_embeddings
        :return: dict
        """
        return ast.literal_eval(s)

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        x_copy[self.col_name] = x_copy[self.col_name].swifter.apply(
            self.normalize_embeddings
        )
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)
