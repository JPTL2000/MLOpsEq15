from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(title="ML Model API", version="1.0")

# Cargar el modelo entrenado
#model = joblib.load("model.joblib")

class InputData(BaseModel):
    x1: float
    x2: float

@app.get("/")
def home():
    return {"message": "🚀 ML Service is running"}

@app.post("/predict")
def predict(data: InputData):
#    X = [[data.x1, data.x2]]
#    prediction = model.predict(X)
#    return {"prediction": int(prediction[0])}
    return {"message": "🚀 ML Service is running"}