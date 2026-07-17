import argparse

import mlflow

from src.models._hf_common import train_and_eval

# General 4-layer distilled BERT, ~14M params vs DistilBERT's ~66M.
# Included specifically to make the size/accuracy/latency tradeoff visible
# in the MLflow comparison rather than just picking DistilBERT by default.
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/reviews.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output-dir", default="artifacts/models/tinybert")
    args = parser.parse_args()

    mlflow.set_tracking_uri("artifacts/mlruns")
    mlflow.set_experiment("reviewguard-sentiment")

    with mlflow.start_run(run_name="tinybert-4l-312d"):
        mlflow.log_param("model_type", "tinybert")
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)

        results = train_and_eval(
            model_name=MODEL_NAME,
            data_path=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("f1_macro", results["f1_macro"])
        mlflow.log_metric("latency_ms_per_review", results["latency_ms"])

        print(f"tinybert accuracy={results['accuracy']:.4f} "
              f"f1_macro={results['f1_macro']:.4f} "
              f"latency={results['latency_ms']:.2f}ms")
        print(f"model saved to {results['model_path']}")


if __name__ == "__main__":
    main()
