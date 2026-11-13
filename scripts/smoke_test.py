"""Run against a STARTED API, including model readiness and inference."""

import os
import sys
import httpx
from sklearn.datasets import load_digits


def main():
    base = os.getenv("API_URL", "http://127.0.0.1:8000")
    with httpx.Client(timeout=10) as client:
        live = client.get(base + "/health/live")
        ready = client.get(base + "/health/ready")
        prediction = client.post(base + "/predict", json={"pixels": load_digits().images[0].tolist()})
    print("liveness:", live.status_code, "readiness:", ready.status_code, "prediction:", prediction.status_code)
    if (live.status_code, ready.status_code, prediction.status_code) != (200, 200, 200):
        sys.exit(1)
    body = prediction.json()
    assert isinstance(body["digit"], int) and 0 <= body["digit"] <= 9
    assert 0 <= body["confidence"] <= 1
    print("Smoke test passed:", body)


if __name__ == "__main__":
    main()
