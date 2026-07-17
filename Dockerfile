FROM python:3.11-slim

WORKDIR /app

# system deps some of the ML libs need for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image a lot smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/

# no checkpoint baked into the image by default - mount one in or set
# REVIEWGUARD_MODEL_DIR to point at a volume, otherwise runs on the
# lexicon fallback (fine for smoke-testing the container itself)
ENV REVIEWGUARD_MODEL_DIR=""
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
