import os
import pytest
from fastapi.testclient import TestClient

import api.inference as inference_module
from api.main import app


@pytest.fixture(autouse=True)
def reset_engine(monkeypatch):
    # force the fallback (no-checkpoint) path so tests don't need network
    # access to pull a model, and reset the module-level singleton each test
    monkeypatch.delenv("REVIEWGUARD_MODEL_DIR", raising=False)
    inference_module._engine = None
    yield
    inference_module._engine = None


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["using_fallback_classifier"] is True


def test_analyze_flags_mismatched_review(client):
    resp = client.post("/analyze", json={
        "rating": 5,
        "text": "Broke after two days, complete waste of money, terrible.",
        "review_id": "r1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["review_id"] == "r1"
    assert body["predicted_sentiment"] == "negative"
    assert body["is_flagged"] is True
    assert body["explanation"] is not None


def test_analyze_does_not_flag_consistent_review(client):
    resp = client.post("/analyze", json={
        "rating": 5,
        "text": "Works great, love it, excellent quality.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_flagged"] is False


def test_analyze_rejects_out_of_range_rating(client):
    resp = client.post("/analyze", json={"rating": 7, "text": "fine"})
    assert resp.status_code == 422  # pydantic validation error


def test_analyze_rejects_empty_text(client):
    resp = client.post("/analyze", json={"rating": 3, "text": ""})
    assert resp.status_code == 422


def test_batch_analyze_returns_result_per_review(client):
    resp = client.post("/analyze/batch", json={
        "reviews": [
            {"rating": 5, "text": "Terrible, broke immediately, waste of money."},
            {"rating": 5, "text": "Works great, love it."},
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert "duplicate_clusters" in body


def test_batch_analyze_rejects_empty_list(client):
    resp = client.post("/analyze/batch", json={"reviews": []})
    assert resp.status_code == 400
