import os

import click
import pandas as pd

from gorgias_ml.contact_reason import ContactReason


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--from-dir", type=str, required=False)
@click.option("--from-models-dir", type=str, required=False)
@click.option("--from-pipelines-dir", type=str, required=False)
@click.option("--df-filename", type=str, required=True)
def predict(
    data_dir: str,
    output_dir: str,
    from_dir: str,
    from_models_dir: str,
    from_pipelines_dir: str,
    df_filename: str,
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

    contact_reason.save_predictions(predictions=predictions, output_dir=output_dir)
