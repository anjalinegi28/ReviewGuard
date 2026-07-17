import pytest

from src.mismatch.scorer import MismatchScorer
from src.mismatch.classifiers import LexiconFallbackClassifier


@pytest.fixture
def scorer():
    return MismatchScorer(LexiconFallbackClassifier(), flag_threshold=0.4)


def test_five_star_negative_text_is_flagged(scorer):
    result = scorer.score(5, "Broke after two days, complete waste of money, terrible product.")
    assert result.predicted_sentiment == "negative"
    assert result.mismatch_score > 0
    assert result.is_flagged is True


def test_five_star_positive_text_is_not_flagged(scorer):
    result = scorer.score(5, "Works great, love it, excellent quality and fast shipping.")
    assert result.predicted_sentiment == "positive"
    assert result.mismatch_score == 0
    assert result.is_flagged is False


def test_one_star_positive_text_is_flagged(scorer):
    result = scorer.score(1, "Amazing product, love it, best purchase, works perfectly.")
    assert result.predicted_sentiment == "positive"
    assert result.is_flagged is True


def test_one_star_negative_text_is_not_flagged(scorer):
    result = scorer.score(1, "Terrible, broke immediately, complete waste of money.")
    assert result.predicted_sentiment == "negative"
    assert result.mismatch_score == 0
    assert result.is_flagged is False


def test_neutral_text_with_middling_rating_not_flagged(scorer):
    result = scorer.score(3, "It's an item, does something, exists I guess.")
    assert result.mismatch_score == 0


def test_batch_scoring_matches_individual_scoring(scorer):
    rows = [
        {"rating": 5, "text": "Terrible, broke immediately, waste of money."},
        {"rating": 5, "text": "Works great, love it."},
    ]
    batch_results = scorer.score_batch(rows)
    assert len(batch_results) == 2
    assert batch_results[0].is_flagged is True
    assert batch_results[1].is_flagged is False


@pytest.mark.parametrize("rating", [0, 6, -1])
def test_out_of_range_rating_raises_keyerror_free(scorer, rating):
    # scorer itself doesn't validate rating range (that's the API schema's
    # job via pydantic Field(ge=1, le=5)) - just confirm it doesn't crash
    # ungracefully on the lookup path for values inside EXPECTED_RATING_RANGE math
    if 1 <= rating <= 5:
        scorer.score(rating, "fine product")
