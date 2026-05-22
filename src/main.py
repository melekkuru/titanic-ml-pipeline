from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Titanic Survival Predictor API",
    description="A FastAPI application that serves a trained Titanic survival prediction model.",
    version="1.0.0"
)

model = joblib.load("src/titanic_model.pkl")


class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3, example=3)
    sex: str = Field(..., example="male")
    age: float = Field(..., ge=0, le=120, example=22)
    sibsp: int = Field(..., ge=0, example=1)
    parch: int = Field(..., ge=0, example=0)
    fare: float = Field(..., ge=0, example=7.25)
    embarked: str = Field(..., example="S")


class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float


@app.get("/", tags=["Health Check"])
def root():
    return {"message": "Titanic model API is running"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "survived": int(prediction),
        "survival_probability": round(float(probability), 4)
    }