import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ReviewIn, BatchReviewsIn, AnalyzeOut, BatchAnalyzeOut
from api.inference import get_engine

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ReviewGuard",
    description="Rating-text mismatch and templated/fake review detector.",
    version="0.1.0",
)

# wide open for a portfolio/demo deployment - lock this down if this ever
# sits in front of anything real
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "using_fallback_classifier": engine.using_fallback,
        "dedup_enabled": engine.dedup is not None,
    }


@app.post("/analyze", response_model=AnalyzeOut)
def analyze(review: ReviewIn):
    engine = get_engine()
    try:
        result = engine.analyze_one(review.rating, review.text, review.review_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
    return result


@app.post("/analyze/batch", response_model=BatchAnalyzeOut)
def analyze_batch(payload: BatchReviewsIn):
    if not payload.reviews:
        raise HTTPException(status_code=400, detail="reviews list is empty")
    engine = get_engine()
    rows = [{"rating": r.rating, "text": r.text, "review_id": r.review_id} for r in payload.reviews]
    try:
        result = engine.analyze_batch(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"batch inference failed: {e}")
    return result
