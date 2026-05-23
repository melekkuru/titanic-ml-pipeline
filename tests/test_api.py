import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.main import app

client = TestClient(app)

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Titanic model API is running"}


def test_predict_valid_input():
    response = client.post(
        "/predict",
        json={
            "pclass": 3,
            "sex": "male",
            "age": 22,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "S",
        },
    )

    assert response.status_code == 200

    response_json = response.json()

    assert "survived" in response_json
    assert "survival_probability" in response_json
    assert response_json["survived"] in [0, 1]
    assert 0 <= response_json["survival_probability"] <= 1


def test_predict_invalid_age():
    response = client.post(
        "/predict",
        json={
            "pclass": 3,
            "sex": "male",
            "age": -5,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "S",
        },
    )

    assert response.status_code == 422


def test_predict_invalid_enum_values():
    response = client.post(
        "/predict",
        json={
            "pclass": 3,
            "sex": "banana",
            "age": 22,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "X",
        },
    )

    assert response.status_code == 422