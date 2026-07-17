"""
This is the actual point of the project: take a star rating and the review
text, predict what the text *actually* sounds like independent of the
rating, and score how far apart they are.

A 5-star review with negative text isn't a sentiment-analysis curiosity,
it's one of the more reliable individual signals for either a bot-generated
review, an incentivized/paid review where the writer forgot to match tone,
or (less excitingly) someone who fat-fingered the star widget. Marketplaces
care about all three.
"""
from dataclasses import dataclass, field
from typing import Optional

# Expected rating range for each predicted sentiment class - anything
# outside this range for a given predicted class counts as a mismatch.
# Ranges overlap at the edges on purpose: a 3-star review with mildly
# positive text isn't really "wrong" the way a 5-star with venomous text is.
EXPECTED_RATING_RANGE = {
    "negative": (1, 2),
    "neutral": (2, 4),
    "positive": (4, 5),
}

SENTIMENT_TO_CENTER_RATING = {"negative": 1.5, "neutral": 3.0, "positive": 4.5}


@dataclass
class MismatchResult:
    rating: int
    predicted_sentiment: str
    sentiment_confidence: float
    mismatch_score: float          # 0 = fully consistent, 1 = maximally inconsistent
    is_flagged: bool
    reason: str


class MismatchScorer:
    """
    Wraps a sentiment classifier (HF pipeline, or anything exposing the same
    .predict(text) -> (label, confidence) interface) and turns its output
    into a mismatch score against the star rating.
    """

    def __init__(self, classifier, flag_threshold: float = 0.35):
        """
        classifier: object with a `.predict(text: str) -> (str, float)` method
                    returning one of "negative"/"neutral"/"positive" and a
                    confidence in [0, 1]. See src/mismatch/classifiers.py for
                    concrete implementations (HF pipeline wrapper + lexicon
                    fallback for tests/dev without a trained checkpoint).
        flag_threshold: mismatch scores >= this get is_flagged=True
        """
        self.classifier = classifier
        self.flag_threshold = flag_threshold

    def score(self, rating: int, text: str) -> MismatchResult:
        sentiment, confidence = self.classifier.predict(text)
        low, high = EXPECTED_RATING_RANGE[sentiment]

        if low <= rating <= high:
            distance = 0.0
        else:
            distance = min(abs(rating - low), abs(rating - high)) / 4.0  # normalize by max possible gap

        # weight the raw distance by how confident the model actually was -
        # a low-confidence "negative" call shouldn't tank a 5-star review's
        # score as hard as a high-confidence one would
        mismatch_score = round(distance * confidence, 3)
        is_flagged = mismatch_score >= self.flag_threshold

        if distance == 0.0:
            reason = f"text sentiment ({sentiment}) is consistent with a {rating}-star rating"
        else:
            reason = (
                f"text reads as {sentiment} (confidence {confidence:.2f}) but rating is "
                f"{rating} stars - expected sentiment for that rating range is different"
            )

        return MismatchResult(
            rating=rating,
            predicted_sentiment=sentiment,
            sentiment_confidence=round(confidence, 3),
            mismatch_score=mismatch_score,
            is_flagged=is_flagged,
            reason=reason,
        )

    def score_batch(self, rows: list) -> list:
        """rows: list of {"rating": int, "text": str}"""
        return [self.score(r["rating"], r["text"]) for r in rows]
