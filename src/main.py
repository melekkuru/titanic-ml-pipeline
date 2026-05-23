from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Titanic Survival Predictor API",
    description="A FastAPI application that serves a trained Titanic survival prediction model.",
    version="1.0.0"
)

MODEL_PATH = Path(__file__).parent / "titanic_model.pkl"
model = joblib.load(MODEL_PATH)

class Sex(str, Enum):
    male = "male"
    female = "female"


class Embarked(str, Enum):
    S = "S"
    C = "C"
    Q = "Q"

class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3, example=3)
    sex: Sex = Field(..., example="male")
    age: float = Field(..., ge=0, le=120, example=22)
    sibsp: int = Field(..., ge=0, example=1)
    parch: int = Field(..., ge=0, example=0)
    fare: float = Field(..., ge=0, example=7.25)
    embarked: Embarked = Field(..., example="S")


class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float

@app.get("/", tags=["Health Check"])
def root():
    return {"message": "Titanic model API is running"}
    
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(passenger: Passenger):
    try:
        data = pd.DataFrame([passenger.model_dump()])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        return {
            "survived": int(prediction),
            "survival_probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )