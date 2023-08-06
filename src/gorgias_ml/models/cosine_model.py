import numpy as np
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
        self.features_col_name = features_col_name
        self.prediction_col_name = prediction_col_name
        self._centroids = None

    @property
    def centroids(self) -> dict[str, list[float]]:
        return self._centroids

    def _predict(self, emb: np.array) -> str:
        """
        It computes the cosine similarity between message and centroid of each class and predicts argmax
        :param emb: embeddings of the message
        :return: class name
        """
        return max(
            self._classes,
            key=lambda cr: sp.spatial.distance.cosine(emb, self._centroids[cr]),
        )

    @staticmethod
    def _compute_mean(arrays: pd.Series) -> np.array:
        return np.mean(np.vstack(arrays.tolist()), axis=0)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        Computes the centroid of each class
        :param x: training dataframe
        :param y: classes
        :return: self
        """
        if y is None:
            ValueError("requires y to be passed, but the target y is None")

        self._classes = y.to_list()

        assert (x.features.apply(lambda z: z.shape[0]) == 384).mean() == 1.0

        self._centroids = (
            x.groupby(self.target_col_name)[self.features_col_name]
            .apply(self._compute_mean)
            .to_dict()
        )

        self.is_fitted_ = True

        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            ValueError("the model is not fitted yet")

        x_copy = x.copy()
        x_copy[self.prediction_col_name] = x_copy[self.features_col_name].swifter.apply(
            lambda emb: self._predict(emb)
        )
        return x_copy
