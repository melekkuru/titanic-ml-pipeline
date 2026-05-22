# Titanic ML Pipeline & FastAPI Deployment

An end-to-end Machine Learning pipeline project built with Scikit-learn and FastAPI for Titanic survival prediction.

This project demonstrates:
- data preprocessing
- feature engineering
- machine learning pipelines
- model comparison
- classification analysis
- model serialization
- ML model deployment with FastAPI
- real-time inference API development

---

# Project Overview

The goal of this project is to predict passenger survival on the Titanic dataset using multiple machine learning models and deploy the trained model as a production-style inference API.

The workflow includes:
- preprocessing structured tabular data
- preventing data leakage with Scikit-learn Pipelines
- comparing classification models
- evaluating model performance
- serializing the trained model with Joblib
- serving predictions through FastAPI

---

# Data Source

The Titanic dataset is loaded directly from Seaborn:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
````

No local dataset file is required to run the notebook.

---

# Project Workflow

```text
Raw Data
↓
EDA
↓
Feature Selection
↓
Missing Value Handling
↓
Encoding
↓
Scaling
↓
Train/Test Split
↓
Model Training
↓
Model Comparison
↓
Model Serialization (.pkl)
↓
FastAPI Deployment
↓
Real-Time Prediction API
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib
* XGBoost
* FastAPI
* Uvicorn
* Pydantic
* Joblib

---

# Machine Learning Concepts Covered

## Data Preprocessing

* Missing value handling
* OneHotEncoder
* StandardScaler
* ColumnTransformer
* Scikit-learn Pipelines
* Data leakage prevention

## Model Training

* Logistic Regression
* Random Forest
* XGBoost

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

# Models Compared

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.804 |     0.793 |  0.667 |    0.724 |   0.843 |
| Random Forest       |    0.810 |     0.787 |  0.696 |    0.738 |   0.840 |
| XGBoost             |    0.804 |     0.758 |  0.724 |    0.741 |   0.809 |

---

# Final Model Evaluation

The Logistic Regression model was selected as the final baseline model because it is simple, interpretable, and achieved strong ROC-AUC performance.

Classification report summary:

| Class               | Precision | Recall | F1-Score | Support |
| ------------------- | --------: | -----: | -------: | ------: |
| 0 - Did not survive |      0.81 |   0.89 |     0.85 |     110 |
| 1 - Survived        |      0.79 |   0.67 |     0.72 |      69 |

Overall accuracy: **0.80**

The model performed better on class `0` than class `1`, meaning it was better at identifying passengers who did not survive than passengers who survived.

---

# Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

# FastAPI Deployment

The trained Titanic survival prediction pipeline was serialized using Joblib and deployed through a FastAPI inference API.

The API supports:

* real-time prediction requests
* JSON request validation with Pydantic
* automatic Swagger/OpenAPI documentation
* probability-based survival prediction

---

# Run the API

```bash
uvicorn src.main:app --reload
```

---

# Swagger Documentation

After starting the server:

```text
http://127.0.0.1:8000/docs
```

---

# Example Request

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

---

# Example Response

```json
{
  "survived": 0,
  "survival_probability": 0.0876
}
```

---

# Key Learning Outcomes

This project helped build understanding of:

* how preprocessing workflows are structured
* why feature engineering matters
* how Scikit-learn Pipelines prevent data leakage
* how to compare multiple classification models
* how to interpret classification metrics
* how to serialize trained ML models
* how to expose ML models through APIs
* how FastAPI handles request validation
* how to design production-style ML inference systems

---

# Repository Structure

```text
titanic-ml-pipeline/
│
├── app/
│   └── main.py
│
├── src/
│   ├── main.py
│   └── titanic_model.pkl
│
├── notebook/
│   └── titanic_pipeline.ipynb
│
├── images/
│   └── confusion_matrix.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Future Improvements

* Hyperparameter tuning with GridSearchCV
* Cross-validation optimization
* Feature importance analysis
* Docker containerization
* Cloud deployment (Render/Railway/AWS)
* Authentication and API security
* Monitoring and logging
* CI/CD integration

---

# Author

Melek Kuru

```
```
