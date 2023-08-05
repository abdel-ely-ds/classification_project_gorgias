import logging
import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from datetime import datetime

from sklearn.pipeline import Pipeline

from gorgias_ml.transfomers.cleaners import EmbeddingsCleaner
from gorgias_ml.transfomers.encoders import TicketMessageAverageEncoder
from gorgias_ml.models.cosine_model import TicketMessageSimilarityBasedClassifier
import gorgias_ml.constants as cst


class ContactReason:
    """
    Class responsible for training and inference
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        model: BaseEstimator = None,
        processing_pipe: Pipeline = None,
    ):
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

    @staticmethod
    def _build_model() -> BaseEstimator:
        return TicketMessageSimilarityBasedClassifier

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
        if df[cst.FEATURES].isna().sum() > 0:
            ValueError("features should not be none")

        df_copy = df.copy()
        x = self._processing_pipe.transform(df_copy)
        predictions = self._model.predict(x)
        return pd.DataFrame.from_dict(
            {cst.ACCOUNT_ID: df[cst.ACCOUNT_ID], cst.PREDICTION: predictions}
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

    @staticmethod
    def save_predictions(
        predictions: pd.DataFrame, output_dir: str = cst.PREDICTIONS_DIRECTORY
    ) -> None:
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")

        predictions.to_csv(
            os.path.join(output_dir, cst.PREDICTION_NAME + f"_{current_datetime}.csv"),
            index=False,
        )

    def load_artifacts(
        self,
        from_dir: str = cst.ARTIFACTS_DIRECTORY,
        from_models_dir: str = cst.MODELS_DIRECTORY,
        from_pipelines_dir: str = cst.PIPELINES_DIRECTORY,
    ) -> None:
        model_path = os.listdir(os.path.join(from_dir, from_models_dir))[0]
        pipeline_path = os.listdir(os.path.join(from_dir, from_pipelines_dir))[0]
        self._model = joblib.load(os.path.join(from_dir, from_models_dir, model_path))
        self._processing_pipe = joblib.load(
            os.path.join(from_dir, from_pipelines_dir, pipeline_path)
        )
