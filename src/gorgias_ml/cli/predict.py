import os

import click
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from gorgias_ml.contact_reason import ContactReason
import gorgias_ml.constants as cst


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--from-dir", type=str, default=cst.ARTIFACTS_DIRECTORY, required=False)
@click.option(
    "--from-models-dir", type=str, default=cst.MODELS_DIRECTORY, required=False
)
@click.option(
    "--from-pipelines-dir", type=str, default=cst.PIPELINES_DIRECTORY, required=False
)
@click.option("--df-filename", type=str, required=True)
@click.option("--score", type=bool, default=False, required=False)
def predict(
    data_dir: str,
    output_dir: str,
    from_dir: str,
    from_models_dir: str,
    from_pipelines_dir: str,
    df_filename: str,
    score: bool = False,
) -> None:
    click.echo(f"Inference started...")

    contact_reason = ContactReason()
    df = pd.read_parquet(os.path.join(data_dir, df_filename))
    contact_reason.load_artifacts(
        from_dir=from_dir,
        from_models_dir=from_models_dir,
        from_pipelines_dir=from_pipelines_dir,
    )
    predictions = contact_reason.predict(df=df)

    if score:
        test_preds = predictions[cst.PREDICTION]
        test_truth = df[cst.TARGET]
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            test_truth, test_preds, average="weighted"
        )

        click.echo(
            f"Weighted precision: {precision:.2f} \n"
            f"Weighted recall: {recall:.2f} \n"
            f"Weighted f1_score: {f1_score:.2f}"
        )

    contact_reason.save_predictions(predictions=predictions, output_dir=output_dir)
