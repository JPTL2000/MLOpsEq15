import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Dict, Any

app = FastAPI(title="Online News Prediction", version="0.1.0", description="This is the endpoint to make a prediction for online news")

# -----------------------------
#  Config
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "online_news_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.pkl"

# -----------------------------
#  Utility Functions
# -----------------------------

@lru_cache()
def load_feature_names(path: Path):
    """Load feature names from metadata pickle and cache the result."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, "rb") as f:
        metadata = pickle.load(f)

    if "feature_names" not in metadata:
        raise KeyError("Metadata file does not contain 'feature_names'")

    return metadata["feature_names"]


@lru_cache()
def load_model(path: Path):
    """Load the ML model with caching."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_single(model, features: Dict[str, Any]):
    """Make a single prediction from a dict of features."""
    df = pd.DataFrame([features])
    prob = model.predict(df)[0]
    return prob


# -----------------------------
#  Dynamic Pydantic Model
# -----------------------------

feature_names = load_feature_names(METADATA_PATH)
input_fields = {name: (float, ...) for name in feature_names}
PredictInput = create_model("PredictInput", **input_fields)


# Output model
class PredictOutput(BaseModel):
    probability: float


# -----------------------------
#  FastAPI Endpoint
# -----------------------------

@app.post("/predict", response_model=PredictOutput)
def predict(item: PredictInput):
    try:
        model = load_model(MODEL_PATH)
        features = item.dict()

        prob = predict_single(model, features)

        return PredictOutput(probability=float(prob))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Hello World!"}