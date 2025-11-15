import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from model_extraction import load_feature_names, load_model, predict_single
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Dict, Any

app = FastAPI(title="Online News Prediction", version="0.1.0", description="This is the endpoint to make a prediction for online news")

# -----------------------------
#  Config
# -----------------------------

with open('../model_metadata.txt', 'r') as file:
        metada_file= file.read()

with open('../model_name.txt', 'r') as file:
        model_file= file.read()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / model_file
METADATA_PATH = BASE_DIR / "models" / metada_file


# -----------------------------
#  Dynamic Pydantic Model
# -----------------------------
model = load_model(MODEL_PATH)
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
        features = item.dict()
        prob = predict_single(model, features)

        return PredictOutput(probability=float(prob))
        

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Hello World!"}