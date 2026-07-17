#!/usr/bin/env bash
set -euo pipefail

# point this at a trained checkpoint dir to use the real model instead of
# the lexicon fallback, e.g.:
#   export REVIEWGUARD_MODEL_DIR=artifacts/models/distilbert-final
PORT="${PORT:-8000}"

uvicorn api.main:app --reload --host 0.0.0.0 --port "$PORT"
