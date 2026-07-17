#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/processed/reviews.csv}"

if [ ! -f "$DATA_PATH" ]; then
  echo "no dataset found at $DATA_PATH, generating synthetic dataset..."
  python -m src.data.synthetic --n 4000 --out "$DATA_PATH"
fi

echo "== training baseline (tf-idf + logreg) =="
python -m src.models.baseline --data "$DATA_PATH"

echo "== training tinybert =="
python -m src.models.train_tinybert --data "$DATA_PATH" --epochs 3

echo "== training distilbert =="
python -m src.models.train_distilbert --data "$DATA_PATH" --epochs 3

echo "== comparison table =="
python -m src.mlflow_tracking

echo "done. run 'mlflow ui --backend-store-uri ./artifacts/mlruns' to browse runs."
