"""Inference API. Loads a prevalidated model; never trains during requests."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Digit Vision Inference", version="1.0.0")


class PredictionInput(BaseModel):
    # Exactly 8 rows of 8 finite pixel values each, in the training range.
    pixels: list[list[float]] = Field(min_length=8, max_length=8)

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, value):
        if any(len(row) != 8 for row in value):
            raise ValueError("Each of 8 rows must contain exactly 8 pixels")
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("Pixels must be finite")
        if np.any((array < 0) | (array > 16)):
            raise ValueError("Pixel values must be in [0, 16]")
        return value


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    # joblib uses pickle: load only a trusted model from the controlled build.
    return joblib.load(path)


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    try:
        load_model()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        model = load_model()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Model not ready") from exc
    array = np.asarray(payload.pixels, dtype=float).reshape(1, 64)
    prediction = int(model.predict(array)[0])
    proba = model.predict_proba(array)[0]
    return {"digit": prediction, "confidence": float(np.max(proba))}
