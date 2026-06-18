# Titanic ML Pipeline

An end-to-end Machine Learning pipeline project for predicting Titanic passenger survival using Scikit-learn, FastAPI, and Docker.

This project demonstrates:

* exploratory data analysis (EDA)
* feature engineering
* preprocessing pipelines
* model comparison
* classification evaluation
* model interpretability
* FastAPI deployment
* API validation and testing
* Docker containerization
* CI automation with GitHub Actions
* public deployment via Hugging Face Spaces

---

# Live Demo

Try the model without any setup:

🔗 **[Titanic Survival Predictor (Hugging Face Spaces)](https://huggingface.co/spaces/melekkuru/titanic-survival-predictor)**

---

# Project Overview

The goal of this project is to build a complete machine learning workflow for binary classification using the Titanic dataset.

The project covers the entire ML lifecycle:

* data exploration
* preprocessing
* model training
* evaluation
* deployment
* API serving
* testing
* containerization
* continuous integration
* public deployment

---

# Data Source

The dataset is loaded directly from Seaborn:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
```

No local CSV file is required.

---

# Project Workflow

```text
Raw Data
↓
Exploratory Data Analysis
↓
Feature Selection
↓
Missing Value Handling
↓
Encoding & Scaling
↓
Train/Test Split
↓
Model Training
↓
Model Comparison
↓
Evaluation
↓
Feature Importance Analysis
↓
Model Serialization
↓
FastAPI Deployment
↓
API Testing
↓
Docker Containerization
↓
CI (GitHub Actions)
↓
Public Deployment (Hugging Face Spaces)
```

---

# Architecture

```mermaid
graph TB
    subgraph Local["💻 Local Development / Docker"]
        direction LR
        U1["User"] --> ST1["Streamlit UI<br/>(localhost:8501)"]
        ST1 -->|HTTP POST| API["FastAPI<br/>(localhost:8000)"]
        API --> M1["titanic_model.pkl"]
        M1 --> API
        API -->|JSON response| ST1
    end

    subgraph Cloud["☁️ Public Demo - Hugging Face Spaces"]
        direction LR
        U2["User"] --> ST2["Streamlit App<br/>(self-contained)"]
        ST2 --> M2["titanic_model.pkl<br/>(loaded directly)"]
        M2 --> ST2
    end

    DEV["Training Pipeline<br/>(Jupyter Notebook)"] -->|joblib.dump| MODEL["Trained Model<br/>(.pkl)"]
    MODEL --> M1
    MODEL --> M2

    GH["GitHub Repository"] -->|git push| CI["GitHub Actions<br/>(CI: pytest)"]
    GH -.->|deploy| Cloud
    GH -.->|build| Local
```

**How to read this diagram:**

- **Local / Docker** — Streamlit and FastAPI run as two separate services and communicate over HTTP, mirroring a real microservice setup.
- **Hugging Face Spaces** — Streamlit loads the model directly, with no separate API layer. This keeps the public demo simple and easy to host.
- Both environments use the **same trained model file**, only the serving architecture differs.
- Every push to GitHub triggers **GitHub Actions**, which runs the test suite before changes are considered stable.

---

# Technologies Used

## Data Science & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Seaborn
* Matplotlib

## Backend & Deployment

* FastAPI
* Uvicorn
* Pydantic
* Joblib
* Docker
* Docker Compose
* Streamlit
* Hugging Face Spaces

## Testing & CI

* Pytest
* HTTPX
* GitHub Actions

---

# Exploratory Data Analysis (EDA)

The notebook includes several exploratory visualizations to understand relationships between features and survival outcomes.

## Survival by Gender

Female passengers had significantly higher survival rates compared to male passengers.

## Age Distribution by Survival

The age distribution analysis shows that survival outcomes varied across all age groups, although younger passengers showed slightly higher survival tendencies.

## Survival by Passenger Class

Passengers in higher ticket classes had substantially higher survival probabilities.

---

# Machine Learning Pipeline

The project uses Scikit-learn Pipelines and ColumnTransformer to ensure clean preprocessing and prevent data leakage.

## Preprocessing Steps

### Numerical Features

* Missing value imputation using median
* Feature scaling using StandardScaler

### Categorical Features

* Missing value imputation using most frequent value
* One-hot encoding using OneHotEncoder

---

# Models Compared

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |     0.80 |      0.79 |   0.67 |     0.72 |    0.84 |
| Random Forest       |     0.81 |      0.79 |   0.70 |     0.74 |    0.84 |
| XGBoost             |     0.80 |      0.76 |   0.72 |     0.74 |    0.81 |

---

# Class Imbalance Handling

To address class imbalance, Logistic Regression was retrained using:

```python
class_weight="balanced"
```

This approach improved minority-class sensitivity and helped the model better detect survival outcomes.

---

# Final Model Evaluation

The Logistic Regression model was selected as the final baseline model because it remained:

* interpretable
* lightweight
* fast
* deployment-friendly

Classification report observations:

* strong overall accuracy
* high precision for non-survival predictions
* lower recall for survival predictions
* evidence of class imbalance effects

---

# FastAPI Deployment

The trained model was deployed using FastAPI.

The API supports:

* prediction requests
* input validation
* response schemas
* automatic Swagger documentation
* error handling

## Example Prediction Request

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

## Example API Response

```json
{
  "survived": 0,
  "survival_probability": 0.0876
}
```

---

# API Validation Features

The API includes:

* enum validation
* numerical range validation
* structured response models
* HTTP exception handling

Example validations:

* `sex` must be `male` or `female`
* `embarked` must be `S`, `C`, or `Q`
* age cannot be negative

---

# Testing & Continuous Integration

The project includes automated API tests using Pytest and FastAPI TestClient.

Test coverage includes:

* root endpoint testing
* valid prediction requests
* invalid input handling
* validation error testing

Run tests locally:

```bash
pytest
```

## GitHub Actions CI

Every push and pull request to `main` automatically triggers a GitHub Actions workflow that:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs the full test suite with `pytest`

This ensures that dependency conflicts or breaking changes are caught before they reach production. The workflow is defined in `.github/workflows/ci.yml`.

---

# Docker

The application is fully containerized using Docker and Docker Compose.

## Run with Docker Compose

```bash
docker-compose up -d
```

Open Swagger Documentation:

```
http://localhost:8000/docs
```

## Stop

```bash
docker-compose down
```

## Run with Docker (without Compose)

```bash
docker build -t titanic-api:1.0 .
docker run -d -p 8000:8000 --name titanic-api titanic-api:1.0
```

---

# Running the API Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start the server

```bash
uvicorn src.main:app --reload
```

## Open Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# Streamlit UI

A simple web interface is available for interacting with the model without using Swagger.

This local version calls the FastAPI service over HTTP (see [Architecture](#architecture)).

## Run the UI

Make sure the API is running first (via Docker or locally), then:

```bash
streamlit run streamlit_app/app.py
```

Open in browser:

```
http://localhost:8501
```

---

# Repository Structure

```text
titanic-ml-pipeline/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebook/
│   └── titanic_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── titanic_model.pkl
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

---

# Key Learning Outcomes

This project helped strengthen understanding of:

* end-to-end ML workflows
* preprocessing pipelines
* classification metrics
* model interpretability
* class imbalance handling
* FastAPI deployment
* API testing
* Docker containerization
* continuous integration with GitHub Actions
* public model deployment with Hugging Face Spaces
* production-oriented ML engineering practices

---

# Future Improvements

* Hyperparameter tuning with GridSearchCV
* Cross-validation optimization
* MLflow experiment tracking
* Advanced monitoring and logging
* Database integration for storing predictions

---

# Author

**Melek Kuru**