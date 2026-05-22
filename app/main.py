from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Passenger(BaseModel):
    age: float
    fare: float
    pclass: int


@app.get("/")
def home():
    return {"message": "Titanic API is running"}


@app.get("/predict")
def predict():
    return {
        "prediction": 1,
        "label": "survivor",
        "probability": 0.85
    }


@app.post("/predict")
def predict_with_input(passenger: Passenger):
    return {
        "input": passenger,
        "prediction": 1,
        "label": "survived",
        "probability": 0.85
    }


@app.get("/square")
def square(number: int):
    return {"result": number ** 2}


@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}