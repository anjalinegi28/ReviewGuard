"""
Not much to this - a couple of helpers for pulling the baseline vs TinyBERT
vs DistilBERT comparison out of MLflow without having to open the UI every
time. Handy for dropping a table straight into a README or a PR description.
"""
import mlflow
import pandas as pd


def get_comparison_table(experiment_name: str = "reviewguard-sentiment",
                          tracking_uri: str = "artifacts/mlruns") -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"no experiment named {experiment_name} found at {tracking_uri}")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    cols = [
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.accuracy",
        "metrics.f1_macro",
        "metrics.latency_ms_per_review",
    ]
    cols = [c for c in cols if c in runs.columns]
    table = runs[cols].rename(columns={
        "tags.mlflow.runName": "run",
        "params.model_type": "model_type",
        "metrics.accuracy": "accuracy",
        "metrics.f1_macro": "f1_macro",
        "metrics.latency_ms_per_review": "latency_ms",
    })
    return table.sort_values("f1_macro", ascending=False)


if __name__ == "__main__":
    print(get_comparison_table().to_string(index=False))
