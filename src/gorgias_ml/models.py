import pandas as pd
import scipy as sp
import swifter
from sklearn.base import BaseEstimator, TransformerMixin

import gorgias_ml.constants as cst


class TicketMessageSimilarityBasedClassifier(BaseEstimator, TransformerMixin):
    """Cosine similarity based classifier"""

    def __init__(
        self,
        target_col_name: str = cst.TARGET,
        features_col_name: str = cst.FEATURES,
        prediction_col_name: str = cst.PREDICTION,
    ):
        self.target_col_name = target_col_name
        self.feature_col_name = features_col_name
        self.prediction_col_name = prediction_col_name
        self._mean_embeddings = None

    def predict(self, emb, targets: list[str]) -> str:
        """
        It computes the cosine similarity between message and centroid of each class and predicts argmax
        :param emb: embeddings of the message
        :param targets: classes to predict
        :return: class name
        """
        return max(
            targets, key=lambda cr: sp.distance.cosine(emb, self._mean_embeddings[cr])
        )

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        """
        Computes the centroid of each class
        :param x: training dataframe
        :param y: classes
        :return: self
        """
        self._mean_embeddings = (
            x.groupby(self.target_col_name)[self.feature_col_name]
            .mean(axis=1)
            .to_dict()
        )
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None):
        x_copy = x.copy()
        y_list = y.to_list()
        x_copy[self.prediction_col_name] = x_copy[self.feature_col_name].swifter.apply(
            lambda emb: self.predict(emb, y_list)
        )
        return x_copy

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None, **fit_params):
        return self.fit(x, y).transform(x, y)
