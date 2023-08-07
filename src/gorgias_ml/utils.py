import click
from sklearn.metrics import precision_recall_fscore_support


def echo_results(precision, recall, f1_score):
    click.echo(
        f"Weighted precision: {precision:.2f} \n"
        f"Weighted recall: {recall:.2f} \n"
        f"Weighted f1_score: {f1_score:.2f}"
    )


def score_model(truth, preds):
    pr, rec, fs, _ = precision_recall_fscore_support(truth, preds, average="weighted")
    return pr, rec, fs