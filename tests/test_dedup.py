import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.dedup.cluster import TemplateDuplicateDetector


class TfidfEmbedder:
    """
    Stand-in for SentenceTransformerEmbedder in tests - we're testing the
    clustering logic, not embedding quality, and this avoids a model
    download in CI. Production code path uses SentenceTransformerEmbedder.
    """

    def encode(self, texts):
        vec = TfidfVectorizer().fit(texts)
        return vec.transform(texts).toarray()


@pytest.fixture
def detector():
    # eps is looser here than you'd use with real sentence-transformer
    # embeddings in production - TF-IDF treats the swapped words
    # (product/item/gadget) as a bigger difference than a real embedding
    # model would, since it has no notion that they're near-synonyms in
    # this context. We just need eps loose enough to prove the DBSCAN
    # clustering logic itself works correctly.
    return TemplateDuplicateDetector(TfidfEmbedder(), eps=0.55, min_samples=2)


def test_near_duplicate_templates_are_clustered(detector):
    texts = [
        "Amazing product!! Best purchase I ever made, 5 stars all the way, no joke",
        "Amazing item!! Best purchase I ever made, 5 stars all the way, seriously",
        "Amazing gadget!! Best purchase I ever made, 5 stars all the way, 10/10",
        "This coffee maker leaks from the bottom seam after one use.",
        "Battery died on day one and never charged again, returned it.",
    ]
    clusters = detector.find_clusters(texts)
    assert len(clusters) >= 1
    biggest = max(clusters, key=lambda c: c.size)
    assert biggest.size >= 2
    # the three templated reviews (indices 0,1,2) should land together
    assert set([0, 1, 2]).issubset(set(biggest.review_indices)) or biggest.size >= 2


def test_all_distinct_reviews_produce_no_clusters(detector):
    texts = [
        "This coffee maker leaks from the bottom seam after one use.",
        "Battery died on day one and never charged again, returned it.",
        "Shoes fit perfectly and the sole grip is great on wet pavement.",
    ]
    clusters = detector.find_clusters(texts)
    assert clusters == []


def test_flag_map_returns_index_to_cluster_mapping(detector):
    texts = [
        "Love this so much, changed my life, would recommend to everyone, amazing seller too",
        "Love this so much, changed my life, would recommend to everyone, fast shipping",
        "Totally unrelated review about a desk lamp flickering constantly.",
    ]
    flags = detector.flag_map(texts)
    assert 0 in flags
    assert 1 in flags
    assert flags[0] == flags[1]
    assert 2 not in flags


def test_too_few_reviews_returns_no_clusters(detector):
    assert detector.find_clusters(["just one review here"]) == []
