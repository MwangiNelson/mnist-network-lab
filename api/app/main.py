from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pixels: list[float] = Field(min_length=784, max_length=784)

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, values: list[float]) -> list[float]:
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Every pixel must be a finite number between 0 and 1.")
        return values


class ClassProbability(BaseModel):
    digit: int
    probability: float


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[ClassProbability]
    model: str


class Predictor:
    def __init__(self, model_path: Path, metadata_path: Path):
        from tensorflow import keras

        self.model = keras.models.load_model(model_path)
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text("utf-8"))
        input_shape = tuple(self.model.input_shape[1:])
        if input_shape != (784,):
            raise ValueError(f"Expected model input shape (784,), found {input_shape}")

    def predict(self, pixels: list[float]) -> PredictionResponse:
        batch = np.asarray(pixels, dtype=np.float32).reshape(1, 784)
        raw = self.model.predict(batch, verbose=0)[0]
        probabilities = raw.astype(float)
        digit = int(np.argmax(probabilities))
        ranked = sorted(
            (
                ClassProbability(digit=index, probability=float(probability))
                for index, probability in enumerate(probabilities)
            ),
            key=lambda item: item.probability,
            reverse=True,
        )
        return PredictionResponse(
            digit=digit,
            confidence=float(probabilities[digit]),
            probabilities=ranked,
            model=str(self.metadata.get("model", "final_fully_connected")),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "/models/final_fc.keras"))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", "/models/model_metadata.json"))
    try:
        app.state.predictor = Predictor(model_path, metadata_path)
        app.state.load_error = None
    except Exception as error:
        app.state.predictor = None
        app.state.load_error = str(error)
    yield


app = FastAPI(
    title="MNIST network lab API",
    version="1.0.0",
    description="Public inference API for the final student-designed fully connected model.",
    lifespan=lifespan,
)

origins = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if app.state.predictor is not None else "degraded",
        "model_ready": app.state.predictor is not None,
        "error": app.state.load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    if max(payload.pixels) < 0.05:
        raise HTTPException(status_code=422, detail="Draw a digit before predicting.")
    return app.state.predictor.predict(payload.pixels)
