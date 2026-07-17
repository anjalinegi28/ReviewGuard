# ReviewGuard

Rating-text mismatch and fake/templated review detector.

Most "review analysis" demos stop at sentiment. That's not the actual pain point
for a marketplace trust & safety team. The thing that costs money is a 5-star
review that reads like a complaint ("broke after two days, waste of money") or
a wall of 1-star reviews that are all secretly the same templated text with
three words swapped out. ReviewGuard flags both.

## What it actually does

1. **Mismatch scoring** — a fine-tuned DistilBERT classifier predicts the
   sentiment of the review body independent of the star rating the user gave.
   If the model is confident the text is negative but the rating is 4-5 stars
   (or vice versa), that's flagged as a mismatch, with a 0-1 severity score.
2. **Near-duplicate / templated review detection** — review text is embedded
   with `sentence-transformers` (`all-MiniLM-L6-v2`), then clustered
   (cosine similarity + DBSCAN) to catch reviews that are near-copies of each
   other, which is one of the more common fake-review patterns on marketplaces.
3. **Explainability** — SHAP is run over the classifier so that when something
   gets flagged, you get back the actual tokens that pushed the prediction
   away from the star rating, not just a bare score.
4. **Model comparison** — logistic regression (TF-IDF) baseline vs TinyBERT vs
   DistilBERT, tracked in MLflow (accuracy, F1, inference latency per review).
5. **Export** — the winning model gets exported to TFLite so it can plausibly
   run client-side (browser extension / mobile), not just server-side.

## Repo layout

```
reviewguard/
├── src/
│   ├── data/            # dataset pulling + cleaning + synthetic fallback
│   ├── models/          # baseline, DistilBERT, TinyBERT, TFLite export
│   ├── mismatch/         # rating vs predicted-sentiment scorer
│   ├── dedup/            # embedding + clustering for templated reviews
│   ├── explain/           # SHAP wrapper
│   └── mlflow_tracking.py
├── api/                 # FastAPI service
├── tests/                # pytest suite
├── scripts/              # one-off shell scripts for training / running
├── data/                 # raw + processed (gitignored past a sample)
├── artifacts/             # trained model weights, MLflow runs (gitignored)
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting data

Real run (needs internet access to Hugging Face Hub):

```bash
python -m src.data.download --source amazon_polarity --limit 20000
```

No internet / quick local test — generates a synthetic but realistic dataset
with deliberate rating/text mismatches and duplicated templates baked in so
the rest of the pipeline has something to catch:

```bash
python -m src.data.synthetic --n 4000 --out data/processed/reviews.csv
```

## Training + comparing models

```bash
python -m src.models.baseline --data data/processed/reviews.csv
python -m src.models.train_distilbert --data data/processed/reviews.csv --epochs 3
python -m src.models.train_tinybert --data data/processed/reviews.csv --epochs 3
```

Each run logs params/metrics/latency to MLflow. Spin up the UI with:

```bash
mlflow ui --backend-store-uri ./artifacts/mlruns
```

Pick the winner and export it:

```bash
python -m src.models.convert_tflite --checkpoint artifacts/models/distilbert-final --out artifacts/models/reviewguard.tflite
```

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "text": "Broke after two days, complete waste of money, do not buy."}'
```

## Docker

```bash
docker build -t reviewguard .
docker run -p 8000:8000 reviewguard
```

## Tests

```bash
pytest -v
```

## Deploying

Two options that cost nothing to keep running for a portfolio project:

- **Hugging Face Spaces** (Docker SDK) — push this repo, point the Space at
  the `Dockerfile`, done.
- **Render** — connect the repo, set the start command to
  `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.

## Known limitations / honest notes

- The DistilBERT/TinyBERT fine-tuning was done on a review-polarity dataset,
  not a marketplace-specific fraud dataset — there isn't a public one, so
  mismatch severity is a proxy signal, not a certified fraud score.
- Dedup clustering flags *near*-duplicates; it will not catch fake reviews
  that are individually well-written and non-repetitive. It's one signal
  among several a real trust & safety pipeline would use.
- TFLite export is included to demonstrate the workflow, not because this
  particular model is small enough to be a great mobile citizen — DistilBERT
  quantized is ~90MB, workable for browser-extension-adjacent use, not ideal
  for a phone app.
