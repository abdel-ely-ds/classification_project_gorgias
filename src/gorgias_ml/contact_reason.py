import logging
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import gorgias_ml.constants as cst
from gorgias_ml.models.model import TicketMessageClassifier, Distances
from gorgias_ml.transfomers.cleaners import EmbeddingsCleaner
from gorgias_ml.transfomers.encoders import TicketMessageAverageEncoder


class ContactReason:
    """
    Class responsible for training and inference
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model: BaseEstimator = None,
        processing_pipe: Pipeline = None,
        centroid_approach: bool = False,
        use_gpu: bool = False,
        distance_metric: Distances = Distances.COSINE_SIMILARITY,
        k: int = 3,
    ):
        self.distance_metric = distance_metric
        self.k = k
        self.use_gpu = use_gpu
        self.centroid_approach = centroid_approach
        self._model = self._build_model() if model is None else model
        self._processing_pipe = (
            self._build_processing_pipe()
            if processing_pipe is None
            else processing_pipe
        )

    @property
    def processing_pipe(self) -> Pipeline:
        return self._processing_pipe

    @property
    def model(self) -> BaseEstimator:
        return self._model

    def _build_model(self) -> BaseEstimator:
        return TicketMessageClassifier(
            centroid_approach=self.centroid_approach,
            use_gpu=self.use_gpu,
            k=self.k,
            distance_metric=self.distance_metric,
        )

    @staticmethod
    def _build_processing_pipe() -> Pipeline:
        pipe = Pipeline(
            [
                ("embeddings_cleaner", EmbeddingsCleaner()),
                ("message_encoder", TicketMessageAverageEncoder()),
            ]
        )
        return pipe

    def fit(self, df: pd.DataFrame) -> None:
        x_train = self._processing_pipe.fit_transform(df)
        self._model.fit(x_train, x_train[cst.TARGET])

        self.logger.info("training finished!")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        x = self._processing_pipe.transform(df_copy)
        x = self._model.predict(x)
        return pd.DataFrame.from_dict(
            {cst.ACCOUNT_ID: x[cst.ACCOUNT_ID], cst.PREDICTION: x[cst.PREDICTION]}
        )

    def save_artifacts(self, output_dir: str = cst.ARTIFACTS_DIRECTORY) -> None:
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")

        joblib.dump(
            self._processing_pipe,
            os.path.join(
                output_dir,
                cst.PIPELINES_DIRECTORY,
                cst.PIPELINE_NAME + f"_{current_datetime}",
            ),
        )

        joblib.dump(
            self._model,
            os.path.join(
                output_dir,
                cst.MODELS_DIRECTORY,
                cst.MODEL_NAME + f"_{current_datetime}",
            ),
        )

        self.logger.info(f"artifacts saved successfully to {output_dir}")

    def save_predictions(
        self, predictions: pd.DataFrame, output_dir: str = cst.PREDICTIONS_DIRECTORY
    ) -> None:
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")

        predictions.to_csv(
            os.path.join(output_dir, cst.PREDICTION_NAME + f"_{current_datetime}.csv"),
            index=False,
        )

        self.logger.info(f"predictions saved successfully to {output_dir}")

    def load_artifacts(
        self,
        from_dir: str = cst.ARTIFACTS_DIRECTORY,
        from_models_dir: str = cst.MODELS_DIRECTORY,
        from_pipelines_dir: str = cst.PIPELINES_DIRECTORY,
    ) -> None:
        model_path = [
            file
            for file in os.listdir(os.path.join(from_dir, from_models_dir))
            if not file.startswith(".")
        ][0]
        pipeline_path = [
            file
            for file in os.listdir(os.path.join(from_dir, from_pipelines_dir))
            if not file.startswith(".")
        ][0]
        self._model = joblib.load(os.path.join(from_dir, from_models_dir, model_path))
        self._processing_pipe = joblib.load(
            os.path.join(from_dir, from_pipelines_dir, pipeline_path)
        )
