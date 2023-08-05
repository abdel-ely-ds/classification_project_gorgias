import pandas as pd
import scipy as sp
import swifter
from sklearn.base import BaseEstimator

import gorgias_ml.constants as cst


class TicketMessageSimilarityBasedClassifier(BaseEstimator):
    """Cosine similarity based classifier"""

    def __init__(
        self,
        target_col_name: str = cst.TARGET,
        features_col_name: str = cst.FEATURES,
        prediction_col_name: str = cst.PREDICTION,
    ):
        self._classes = None
        self.is_fitted_ = False
        self.target_col_name = target_col_name
        self.feature_col_name = features_col_name
        self.prediction_col_name = prediction_col_name
        self._centroids = None

    @property
    def centroids(self) -> dict[str, list[float]]:
        return self._centroids

    def _predict(self, emb) -> str:
        """
        It computes the cosine similarity between message and centroid of each class and predicts argmax
        :param emb: embeddings of the message
        :return: class name
        """
        return max(
            self._classes, key=lambda cr: sp.distance.cosine(emb, self._centroids[cr])
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Computes the centroid of each class
        :param x: training dataframe
        :param y: classes
        :return: self
        """
        if y is None:
            ValueError('requires y to be passed, but the target y is None')

        self._classes = y.to_list()

        self._centroids = (
            x.groupby(self.target_col_name)[self.feature_col_name]
            .mean(axis=1)
            .to_dict()
        )
        self.is_fitted_ = True

        return self

    def predict(self, x: pd.DataFrame):
        x_copy = x.copy()
        x_copy[self.prediction_col_name] = x_copy[self.feature_col_name].swifter.apply(
            lambda emb: self._predict(emb)
        )
        return x_copy