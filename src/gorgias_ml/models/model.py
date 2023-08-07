from collections import Counter

import numpy as np
import pandas as pd
import swifter
from sklearn.base import BaseEstimator
import faiss
import gorgias_ml.constants as cst

import enum


class Distances(enum.Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    L2 = "l2"


class TicketMessageClassifier(BaseEstimator):
    """
    This classifier a mix of knn and kmeans.

    If centroid_approach is set, we compute the centroid of each class in the training data.
    then for inference we compute the most similar or nearst neighbors (depending) on the metric
    and use majority voting.
    """

    def __init__(
        self,
        centroid_approach: bool = False,
        distance_metric: Distances = Distances.COSINE_SIMILARITY,
        k: int = 5,
        use_gpu: bool = False,
        target_col_name: str = cst.TARGET,
        features_col_name: str = cst.FEATURES,
        prediction_col_name: str = cst.PREDICTION,
    ):
        self.centroid_approach = centroid_approach
        self.distance_metric = distance_metric
        self.k = k
        self.use_gpu = use_gpu
        self.target_col_name = target_col_name
        self.features_col_name = features_col_name
        self.prediction_col_name = prediction_col_name

        self._classes = None
        self.is_fitted_ = False
        self._centroids = None
        self._index = None

    @property
    def centroids(self) -> dict[str, list[float]]:
        return self._centroids

    @staticmethod
    def _compute_mean(arrays: pd.Series) -> np.array:
        return np.mean(np.vstack(arrays.tolist()), axis=0)

    def _set_index(self, all_embeddings: np.array) -> None:
        all_embeddings = all_embeddings.astype("float32")
        if self.distance_metric == Distances.COSINE_SIMILARITY:
            faiss.normalize_L2(all_embeddings)
            self._index = (
                faiss.GpuIndexFlatIP(all_embeddings.shape[1])
                if self.use_gpu
                else faiss.IndexFlatIP(all_embeddings.shape[1])
            )

        else:
            self._index = faiss.IndexFlatL2(all_embeddings.shape[1])

        self._index.add(all_embeddings)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """
        :param x: training dataframe
        :param y: classes
        :return: self
        """
        if y is None:
            ValueError("requires y to be passed, but the target y is None")

        self._classes = y.to_list()

        if self.centroid_approach:
            self._centroids = (
                x.groupby(self.target_col_name)[self.features_col_name]
                .apply(self._compute_mean)
                .to_dict()
            )
        all_embeddings = np.vstack(
            list(self._centroids.values())
            if self.centroid_approach
            else x[self.features_col_name].tolist()
        )
        self._set_index(all_embeddings)

        self.is_fitted_ = True

        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            ValueError("the model is not fitted yet")

        x_copy = x.copy()

        query_embeddings = np.vstack(x_copy[self.features_col_name].tolist()).astype(
            "float32"
        )
        faiss.normalize_L2(query_embeddings)

        _, nn_indices = self._index.search(query_embeddings, self.k)

        x_copy[self.prediction_col_name] = [
            self._majority_vote(nn_indices_row) for nn_indices_row in nn_indices
        ]
        return x_copy

    def _majority_vote(self, nn_indices_row):
        """
        Performs majority vote to get the final prediction
        :param nn_indices_row: Indices of the nearest neighbors for a single data point
        :return: Predicted class label based on majority vote
        """
        nearest_classes = [self._classes[i] for i in nn_indices_row]
        counter = Counter(nearest_classes)
        majority_class = counter.most_common(1)[0][0]
        return majority_class
