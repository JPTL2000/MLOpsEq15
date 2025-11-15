from pathlib import Path
import pickle
import pandas as pd
from typing import Dict, Any
import joblib



def load_feature_names(path: Path):
    """Load feature names from metadata pickle and cache the result."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, "rb") as f:
        metadata = pickle.load(f)

    if "feature_names" not in metadata:
        raise KeyError("Metadata file does not contain 'feature_names'")

    return metadata["feature_names"]

def load_model(path: Path):
    """Load the ML model with caching."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        return joblib.load(f)
    
def predict_single(model, features: Dict[str, Any]):
    """Make a single prediction from a dict of features."""
    df = pd.DataFrame([features])
    prob = model.predict(df)[0]
    return prob