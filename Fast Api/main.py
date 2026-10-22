import pickle
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Diabetes Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# main.py ke saath wali folder mein se model load (path ka masla khatam)
MODEL_PATH = Path(__file__).parent / "Mymdoel.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Model ke apne features aur unka order
FEATURES = list(model.feature_names_in_)

class PredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "Diabetes API chal rahi hai. /docs kholo."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])[FEATURES]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "predicted_label": pred,
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability": round(prob, 3),
    }
    
    #python -m uvicorn main:app --reload
    #python -m pip install fastapi uvicorn pandas scikit-learn
#python -m uvicorn main:app --reload --port 8001