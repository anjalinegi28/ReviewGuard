"""
Dumbest thing that could possibly work: TF-IDF + logistic regression.
This exists so the MLflow comparison table has an honest floor to compare
DistilBERT/TinyBERT against - if a transformer can't beat this by a
meaningful margin, that's worth knowing.
"""
import argparse
import time

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_and_clean, LABEL_NAMES


def run(data_path: str, artifacts_dir: str = "artifacts/models"):
    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    mlflow.set_tracking_uri("artifacts/mlruns")
    mlflow.set_experiment("reviewguard-sentiment")

    with mlflow.start_run(run_name="tfidf-logreg-baseline"):
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train_vec, y_train)

        # latency: single-row inference, averaged, since that's the real
        # per-request cost once this sits behind an API
        sample_rows = X_test.iloc[:100]
        t0 = time.perf_counter()
        for row in sample_rows:
            _ = clf.predict(vectorizer.transform([row]))
        latency_ms = (time.perf_counter() - t0) / len(sample_rows) * 1000

        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model_type", "tfidf_logreg")
        mlflow.log_param("max_features", 20000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("latency_ms_per_review", latency_ms)

        print(f"baseline accuracy={acc:.4f} f1_macro={f1:.4f} latency={latency_ms:.2f}ms")

        import os
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(vectorizer, f"{artifacts_dir}/baseline_vectorizer.joblib")
        joblib.dump(clf, f"{artifacts_dir}/baseline_logreg.joblib")
        mlflow.log_artifact(f"{artifacts_dir}/baseline_vectorizer.joblib")
        mlflow.log_artifact(f"{artifacts_dir}/baseline_logreg.joblib")

    return {"accuracy": acc, "f1_macro": f1, "latency_ms": latency_ms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/reviews.csv")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    args = parser.parse_args()
    run(args.data, args.artifacts_dir)
