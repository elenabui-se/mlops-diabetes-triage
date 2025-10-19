# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import load_artifacts, predict as predict_fn

app = FastAPI(title="Diabetes Triage API", version="v0.1")

# load model + scaler at startup
try:
    model, scaler = load_artifacts()
    print("Model and scaler loaded.")
except Exception as e:
    model = scaler = None
    print(f"Warning: could not load artifacts: {e}")

class DiabetesFeatures(BaseModel):
    age: float = Field(..., description="Normalized age")
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.1"}

@app.post("/predict")
def predict_endpoint(features: DiabetesFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    try:
        value = predict_fn(features.dict(), model, scaler)
        return {"prediction": value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")