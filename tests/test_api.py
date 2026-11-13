import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from vision_mlops import api


class FakeModel:
    def predict(self, values):
        return np.array([3])

    def predict_proba(self, values):
        values = np.zeros((1, 10))
        values[0, 3] = 1
        return values


def test_liveness():
    with TestClient(api.app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}


def test_predict_contract(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    pixels = load_digits().images[0].tolist()
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"pixels": pixels})
    assert response.status_code == 200
    assert response.json() == {"digit": 3, "confidence": 1.0}


def test_rejects_invalid_pixels():
    with TestClient(api.app) as client:
        r1 = client.post("/predict", json={"pixels": [[0] * 8] * 7})
        r2 = client.post("/predict", json={"pixels": [[20] * 8] * 8})
    assert r1.status_code == 422
    assert r2.status_code == 422


def test_readiness_requires_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    api.load_model.cache_clear()
    with TestClient(api.app) as client:
        assert client.get("/health/ready").status_code == 503
    api.load_model.cache_clear()
