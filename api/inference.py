"""
Wires the mismatch scorer, dedup detector, and explainer together for the
API layer. Everything here tries to load the "real" trained
model/embedder/explainer first (env var REVIEWGUARD_MODEL_DIR pointing at a
fine-tuned checkpoint), and falls back to the lightweight lexicon versions
if no checkpoint is configured - so the API is demoable out of the box
without requiring a multi-hour training run first, and CI/tests don't need
network access to the Hub.
"""
import os
import logging

from src.mismatch.scorer import MismatchScorer
from src.mismatch.classifiers import HFPipelineClassifier, LexiconFallbackClassifier
from src.dedup.cluster import TemplateDuplicateDetector, SentenceTransformerEmbedder
from src.explain.shap_explain import ShapExplainer, LexiconFallbackExplainer

logger = logging.getLogger("reviewguard")


class ReviewGuardEngine:
    def __init__(self):
        model_dir = os.getenv("REVIEWGUARD_MODEL_DIR")
        self.using_fallback = model_dir is None

        if model_dir:
            try:
                classifier = HFPipelineClassifier(model_dir)
                logger.info(f"loaded fine-tuned classifier from {model_dir}")
            except Exception as e:
                logger.warning(f"failed to load model from {model_dir} ({e}), falling back to lexicon scorer")
                classifier = LexiconFallbackClassifier()
                self.using_fallback = True
        else:
            logger.warning("REVIEWGUARD_MODEL_DIR not set, using lexicon fallback classifier "
                            "(fine for demoing the API, not a substitute for the trained model)")
            classifier = LexiconFallbackClassifier()

        self.scorer = MismatchScorer(classifier)

        try:
            embedder = SentenceTransformerEmbedder()
            logger.info("loaded sentence-transformers embedder for dedup")
        except Exception as e:
            logger.warning(f"sentence-transformers unavailable ({e}), dedup clustering disabled")
            embedder = None
        self.dedup = TemplateDuplicateDetector(embedder) if embedder else None

        self.explainer = None
        if not self.using_fallback:
            try:
                from transformers import pipeline as hf_pipeline
                pipe = hf_pipeline("text-classification", model=model_dir, tokenizer=model_dir, top_k=None)
                self.explainer = ShapExplainer(pipe)
            except Exception as e:
                logger.warning(f"SHAP explainer unavailable ({e}), using lexicon fallback explainer")
                self.explainer = LexiconFallbackExplainer()
        else:
            self.explainer = LexiconFallbackExplainer()

    def analyze_one(self, rating: int, text: str, review_id: str = None) -> dict:
        result = self.scorer.score(rating, text)
        explanation = None
        explanation_method = None
        if result.is_flagged and self.explainer is not None:
            exp = self.explainer.explain(text, result.predicted_sentiment)
            explanation = [{"token": t.token, "weight": t.weight} for t in exp.top_tokens]
            explanation_method = exp.method

        return {
            "review_id": review_id,
            "rating": result.rating,
            "predicted_sentiment": result.predicted_sentiment,
            "sentiment_confidence": result.sentiment_confidence,
            "mismatch_score": result.mismatch_score,
            "is_flagged": result.is_flagged,
            "reason": result.reason,
            "explanation": explanation,
            "explanation_method": explanation_method,
        }

    def analyze_batch(self, reviews: list) -> dict:
        results = [
            self.analyze_one(r["rating"], r["text"], r.get("review_id"))
            for r in reviews
        ]

        duplicate_clusters = []
        if self.dedup is not None and len(reviews) >= 2:
            texts = [r["text"] for r in reviews]
            clusters = self.dedup.find_clusters(texts)
            duplicate_clusters = [
                {
                    "cluster_id": c.cluster_id,
                    "review_indices": c.review_indices,
                    "size": c.size,
                    "representative_text": c.representative_text,
                }
                for c in clusters
            ]

        return {"results": results, "duplicate_clusters": duplicate_clusters}


_engine = None


def get_engine() -> ReviewGuardEngine:
    global _engine
    if _engine is None:
        _engine = ReviewGuardEngine()
    return _engine
