from pathlib import Path

import pytest


def test_app_module_has_required_routes():
    pytest.importorskip("faiss")

    from app.main import app

    route_paths = {
        r.path
        for r in app.routes
    }

    assert "/predict" in route_paths
    assert "/health" in route_paths
    assert "/categories" in route_paths


def _client():
    pytest.importorskip("faiss")
    pytest.importorskip("httpx")

    from src.config import MODELS_DIR
    if not (MODELS_DIR / "classifier.pkl").exists():
        pytest.skip("trained models not present — run train.py first")

    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_predict_returns_full_contract():
    with _client() as client:
        resp = client.post(
            "/predict",
            json={"text": "Backup job failed overnight; disk array at 94% capacity."},
        )
        assert resp.status_code == 200
        body = resp.json()

        # Core routing fields
        assert body["category"] in {
            "Infrastructure", "Access Management", "Storage", "HR Support",
            "Procurement", "Internal Project", "General IT",
        }
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["escalate"], bool)
        # escalate must be consistent with the confidence gate
        assert body["escalate"] == (body["confidence"] < body["threshold_used"])

        # Calibrated class probabilities sum to ~1
        assert abs(sum(body["class_probs"].values()) - 1.0) < 1e-3

        # OOD fields are wired through
        assert isinstance(body["ood"], bool)
        assert 0.0 <= body["entropy"] <= 1.0
        assert isinstance(body["ood_reasons"], list)


def test_predict_flags_out_of_distribution_input():
    with _client() as client:
        body = client.post(
            "/predict",
            json={"text": "I would like a vegetarian lunch with extra guacamole please."},
        ).json()
        # A food order resembles nothing in the IT corpus → should be flagged OOD.
        assert body["ood"] is True
        assert body["ood_reasons"]