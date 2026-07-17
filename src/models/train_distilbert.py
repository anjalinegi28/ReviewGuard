import argparse

import mlflow

from src.models._hf_common import train_and_eval

MODEL_NAME = "distilbert-base-uncased"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/reviews.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="artifacts/models/distilbert")
    args = parser.parse_args()

    mlflow.set_tracking_uri("artifacts/mlruns")
    mlflow.set_experiment("reviewguard-sentiment")

    with mlflow.start_run(run_name="distilbert-base-uncased"):
        mlflow.log_param("model_type", "distilbert")
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

        print(f"distilbert accuracy={results['accuracy']:.4f} "
              f"f1_macro={results['f1_macro']:.4f} "
              f"latency={results['latency_ms']:.2f}ms")
        print(f"model saved to {results['model_path']}")


if __name__ == "__main__":
    main()
