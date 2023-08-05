import os

import click
import pandas as pd

from gorgias_ml.contact_reason import ContactReason


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
def train(
    data_dir: str,
    output_dir: str,
    df_filename: str,
) -> None:
    click.echo(f"Training started...")

    contact_reason = ContactReason()
    df = pd.read_parquet(os.path.join(data_dir, df_filename))

    contact_reason.fit(df=df)

    contact_reason.save_artifacts(output_dir=output_dir)
