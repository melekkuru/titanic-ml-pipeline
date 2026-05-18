from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Passenger(BaseModel):
    age: float
    fare: float
    pclass: int

@app.get(f"/")
def home():
    return {"message": "Titanic API is running"}

@app.get(f"/predict")
def predict():
    return {
        "prediciton":1,
        "label": "survivor",
        "probability": 0.85
    }

@app.post(f"/predict")
def predict_with_input(passenger: Passenger):
    return{
        "input":passenger,
        "prediction": 1,
        "label": "survived",
        "probability": 0.85
    }