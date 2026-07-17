"""
Two implementations of the `.predict(text) -> (label, confidence)` interface
that MismatchScorer expects.

- HFPipelineClassifier: wraps a fine-tuned checkpoint via
  transformers.pipeline. This is what runs in production/the API.
- LexiconFallbackClassifier: a dumb word-list scorer with no dependencies
  and no trained weights required. It exists purely so unit tests and local
  dev work before you've trained anything, and so CI doesn't need to
  download model weights just to test the mismatch math. Do not use this
  for anything that needs to be actually accurate.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import re
from typing import Tuple

LABEL_ORDER = ["negative", "neutral", "positive"]

_POSITIVE_WORDS = {
    "great", "good", "love", "excellent", "amazing", "happy", "perfect",
    "best", "recommend", "sturdy", "fast", "easy", "works", "quality",
}
_NEGATIVE_WORDS = {
    "broke", "broken", "waste", "terrible", "awful", "bad", "worst",
    "defective", "damaged", "refund", "junk", "stopped", "cheap",
    "disappointed", "horrible", "died", "return",
}


class HFPipelineClassifier:
    def __init__(self, model_dir: str, device: int = -1):
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            device=device,
            top_k=None,
        )
        # model was trained with label ids 0/1/2 = negative/neutral/positive,
        # see src/data/preprocess.py:rating_to_sentiment_label
        self._id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text: str) -> Tuple[str, float]:
        scores = self._pipe(text, truncation=True, max_length=128)[0]
        best = max(scores, key=lambda s: s["score"])
        label_id = int(best["label"].split("_")[-1]) if "_" in best["label"] else None
        label = self._id_to_label.get(label_id, best["label"].lower())
        return label, float(best["score"])


class LexiconFallbackClassifier:
    """Word-count sentiment, zero ML, zero downloads. Dev/test use only."""

    def predict(self, text: str) -> Tuple[str, float]:
        words = re.findall(r"[a-z']+", text.lower())
        pos_hits = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg_hits = sum(1 for w in words if w in _NEGATIVE_WORDS)

        if pos_hits == 0 and neg_hits == 0:
            return "neutral", 0.5

        total = pos_hits + neg_hits
        if pos_hits > neg_hits:
            confidence = 0.5 + 0.5 * (pos_hits - neg_hits) / total
            return "positive", min(confidence, 0.99)
        if neg_hits > pos_hits:
            confidence = 0.5 + 0.5 * (neg_hits - pos_hits) / total
            return "negative", min(confidence, 0.99)
        return "neutral", 0.5
