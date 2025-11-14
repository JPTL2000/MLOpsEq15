from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
import os
import pickle
import joblib
import pandas as pd

app = FastAPI(
    title="Online News - Prediction",
    description="Endpoint to serve the trained Online News model.",
    version="0.1.0",
)


class PredictInput(BaseModel):
    features: Dict[str, Any] = Field(..., description="Mapping of feature name to value")
    user: Optional["User"] = Field(None, description="Optional user metadata")


class PredictOutput(BaseModel):
    probability: float
    user_id: Optional[int] = None
    user_name: Optional[str] = None


class User(BaseModel):
    id: int
    name: str
    age: int


def _repo_root() -> Path:
    # src/api -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def read_model_name() -> Optional[str]:
    # model name is stored in `src/model_name.txt`
    possible = [
        _repo_root() / "src" / "model_name.txt",
        _repo_root() / "model_name.txt",
    ]
    for p in possible:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                return None
    return None


def find_model_file(model_name: str) -> Optional[Path]:
    repo = _repo_root()
    candidates = [
        repo / "models" / model_name,
        repo / model_name,
        repo / "src" / model_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_model(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    # try joblib first (works for sklearn pipelines), then pickle
    try:
        return joblib.load(p)
    except Exception:
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            raise RuntimeError(f"Could not load model from {p}: {e}")


def predict_single(model, features: Dict[str, Any]):
    # convert dict to single-row DataFrame (preserves column names)
    if isinstance(features, pd.DataFrame):
        X = features
    else:
        X = pd.DataFrame([features])

    # If model implements predict_proba, use probability of positive class
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # handle binary or multiclass: try to pick class 1 when present
        if probs.shape[1] == 1:
            # some models return a single column
            return float(probs.ravel()[0])
        elif probs.shape[1] >= 2:
            return float(probs[:, 1][0])
        else:
            return float(probs.ravel()[0])

    # fallback to predict
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return float(preds[0])

    raise RuntimeError("Model has neither predict_proba nor predict method")


# Determine model path: allow override with env var, otherwise read model_name.txt
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
model = None
try:
    if MODEL_PATH_ENV:
        model_path = Path(MODEL_PATH_ENV)
    else:
        model_name = read_model_name()
        if not model_name:
            raise FileNotFoundError("Could not determine model name from src/model_name.txt")
        model_file = find_model_file(model_name)
        if model_file is None:
            raise FileNotFoundError(f"Model file {model_name} not found in repository")
        model_path = model_file

    model = load_model(model_path)
except Exception as e:
    # Defer raising until a request comes in; keep startup quiet but log message
    model = None
    _load_error = str(e)


@app.post("/predict", response_model=PredictOutput, tags=["Predictions"])
def predict(item: PredictInput):
    global model
    if model is None:
        detail = globals().get("_load_error", "Model not loaded")
        raise HTTPException(status_code=500, detail=f"Model unavailable: {detail}")

    try:
        prob = predict_single(model, item.features)
        response = {"probability": float(prob)}
        if getattr(item, "user", None):
            response["user_id"] = item.user.id
            response["user_name"] = item.user.name
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Hello World"])
async def home():
    return {"message": "Welcome to Online predictions"}