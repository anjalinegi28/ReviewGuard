# 🛡️ ReviewGuard

**Rating/text mismatch and templated fake-review detector for marketplace platforms.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sentiment analysis tells you what a review says. It doesn't tell you whether that matches the star rating next to it — and that gap is where a lot of review fraud hides. A user taps ⭐⭐⭐⭐⭐ out of habit, then writes "broke after two days, waste of money." A coordinated fake-review campaign doesn't write from scratch — it edits the same template dozens of times. ReviewGuard catches both.

## ✨ What it does

🎯 **Mismatch scoring** — a fine-tuned DistilBERT classifier predicts review sentiment independently of the star rating, then flags disagreement with a continuous severity score rather than a binary flag.

🔍 **Templated-review detection** — review text is embedded with sentence-transformers (all-MiniLM-L6-v2) and clustered via DBSCAN to surface near-duplicate reviews across a batch.

💡 **Explainability** — SHAP values expose the exact tokens driving a flagged prediction, so results are auditable, not a black box.

📊 **Model benchmarking** — TF-IDF + logistic regression baseline vs. TinyBERT vs. DistilBERT, every run logged to MLflow for direct comparison.

📱 **Edge-ready export** — the winning model exports to TFLite for client-side use (browser extension or on-device).

## 🏗️ How it's organized

Think of the project in three layers: **data → models → serving.**

```
reviewguard/
├── src/
│   ├── data/
│   ├── models/
│   ├── mismatch/
│   ├── dedup/
│   ├── explain/
│   └── mlflow_tracking.py
├── api/
├── frontend/
├── tests/
└── scripts/
```

| Folder | What lives here |
|---|---|
| 📥 `src/data/` | Downloads and cleans review data, or generates realistic fake data if no dataset is available |
| 🧠 `src/models/` | The three models being compared — a simple baseline, TinyBERT, and DistilBERT |
| ⚖️ `src/mismatch/` | Compares predicted sentiment against the star rating |
| 🔗 `src/dedup/` | Groups similar reviews together to catch copy-paste fakes |
| 💬 `src/explain/` | Explains *why* a review got flagged |
| 🌐 `api/` | The FastAPI server that ties everything together |
| 🖥️ `frontend/` | A simple web page to try it out |
| ✅ `tests/` | Automated checks that everything still works |

## 🧰 Built with

| Purpose | Tools |
|---|---|
| Backend | Python, FastAPI |
| Machine learning | PyTorch, HuggingFace Transformers (DistilBERT, TinyBERT) |
| Similarity detection | sentence-transformers, scikit-learn (DBSCAN) |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Mobile/edge deployment | TensorFlow Lite |
| Packaging | Docker |
| Automation | GitHub Actions |


## 🚀 Try it yourself

**Step 1 — set up your environment**
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2 — get some data to test with** (no real dataset needed, this generates realistic sample data)
```
python -m src.data.synthetic --n 4000 --out data/processed/reviews.csv
```

**Step 3 — train a model**
```
python -m src.models.train_distilbert --data data/processed/reviews.csv --epochs 3
```

**Step 4 — start the server**
```
uvicorn api.main:app --reload --port 8000
```

**Step 5 — test it with a real request**
```
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"rating": 5, "text": "Broke after two days, complete waste of money, do not buy."}'
```

Prefer Docker over setting up Python locally? Skip straight to the section below.

## 🐳 Run with Docker

One command, no local setup needed:
```
docker build -t reviewguard .
docker run -p 8000:8000 reviewguard
```
The API will be live at `http://localhost:8000`.

## ✅ Run the tests

Make sure everything works before you rely on it:
```
pytest -v
```

## ☁️ Put it online

Both options below are free and take a few minutes:

🤗 **Hugging Face Spaces** — create a new Space, choose "Docker" as the SDK, and point it at this repo's `Dockerfile`. Done.

🎨 **Render** — connect this repo, and set the start command to:
```
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

## 📄 License

MIT — free to use, modify, and share.

## Author
Anjali Negi
