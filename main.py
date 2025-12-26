from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# =========================
# App
# =========================
app = FastAPI(title="Predictive Maintenance API")

# =========================
# Load model + threshold
# =========================
model = joblib.load("model.joblib")

with open("threshold.txt") as f:
    THRESHOLD = float(f.read())

# =========================
# Input schema (5 sensors)
# =========================
class SensorInput(BaseModel):
    sensor14: float
    sensor9: float
    sensor4: float
    sensor7: float
    sensor12: float

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict_failure(data: SensorInput):

    X = np.array([[
        data.sensor14,
        data.sensor9,
        data.sensor4,
        data.sensor7,
        data.sensor12
    ]])

    probability = model.predict_proba(X)[0][1]

    prediction = "Failure Likely" if probability >= THRESHOLD else "Safe"

    return {
        "prediction": prediction,
        "failure_probability": round(float(probability), 4),
        "threshold_used": round(float(THRESHOLD), 4)
    }

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"status": "API running"}
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)