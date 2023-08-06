import os

import click
import pandas as pd

from gorgias_ml.contact_reason import ContactReason
from sklearn.metrics import precision_recall_fscore_support
import gorgias_ml.constants as cst


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--use-gpu", type=bool, default=False, required=False)
@click.option("--centroid-approach", type=bool, default=False, required=False)
@click.option("--score", type=bool, default=True, required=False)
def train(
    data_dir: str,
    output_dir: str,
    df_filename: str,
    use_gpu: bool = False,
    centroid_approach: bool = False,
    score: bool = True,
) -> None:
    click.echo(f"Training started...")

    contact_reason = ContactReason(centroid_approach=centroid_approach, use_gpu=use_gpu)
    df = pd.read_parquet(os.path.join(data_dir, df_filename))
    df = df.sample(1000)

    contact_reason.fit(df=df)

    if score:
        train_preds = contact_reason.predict(df)[cst.PREDICTION]
        train_truth = df[cst.TARGET]
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            train_truth, train_preds, average="weighted"
        )
        click.echo(
            f"Weighted precision: {precision:.2f} \n"
            f"Weighted recall: {recall:.2f} \n"
            f"Weighted f1_score: {f1_score:.2f}"
        )

    contact_reason.save_artifacts(output_dir=output_dir)
