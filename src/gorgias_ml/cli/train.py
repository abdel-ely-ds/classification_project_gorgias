import os

import click
import pandas as pd

from gorgias_ml.contact_reason import ContactReason
from sklearn.metrics import precision_recall_fscore_support
import gorgias_ml.constants as cst
from sklearn.model_selection import train_test_split
from gorgias_ml.models.model import Distances


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--use-gpu", type=bool, default=False, required=False)
@click.option("--centroid-approach", type=bool, default=False, required=False)
@click.option("--score", type=bool, default=True, required=False)
@click.option("--split", type=bool, default=True, required=False)
@click.option("--k", type=int, default=3, required=False)
@click.option(
    "--distance-metric", type=str, default="cosine_similarity", required=False
)
def train(
    data_dir: str,
    output_dir: str,
    df_filename: str,
    use_gpu: bool = False,
    centroid_approach: bool = False,
    score: bool = True,
    split: bool = True,
    k: int = 3,
    distance_metric: str = "cosine_similarity",
) -> None:
    click.echo(f"Training started...")

    contact_reason = ContactReason(
        centroid_approach=centroid_approach,
        use_gpu=use_gpu,
        k=k,
        distance_metric=Distances[distance_metric.upper()],
    )
    df = pd.read_parquet(os.path.join(data_dir, df_filename))

    x_train = df
    x_val = None

    if split:
        x_train, x_val = train_test_split(df, random_state=2023, test_size=0.2)
        x_train.to_parquet(f"{data_dir}/x_train", index=False)
        x_val.to_parquet(f"{data_dir}/x_val", index=False)

    contact_reason.fit(df=x_train)

    if score:
        train_preds = contact_reason.predict(x_train)[cst.PREDICTION]
        train_truth = x_train[cst.TARGET]
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            train_truth, train_preds, average="weighted"
        )
        click.echo(
            "-----------Training Scores-------------\n"
            f"Weighted precision: {precision:.2f} \n"
            f"Weighted recall: {recall:.2f} \n"
            f"Weighted f1_score: {f1_score:.2f} \n"
            "-------------------------------------------"
        )

        if x_val is not None:
            val_preds = contact_reason.predict(x_val)[cst.PREDICTION]
            val_truth = x_val[cst.TARGET]
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                val_truth, val_preds, average="weighted"
            )
            click.echo(
                "-----------Validation Scores--------------\n"
                f"Weighted precision: {precision:.2f} \n"
                f"Weighted recall: {recall:.2f} \n"
                f"Weighted f1_score: {f1_score:.2f} \n"
                "-------------------------------------------"
            )

    contact_reason.save_artifacts(output_dir=output_dir)
