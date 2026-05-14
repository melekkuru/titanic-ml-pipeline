# Titanic ML Pipeline

An end-to-end Machine Learning pipeline project built with Scikit-learn for Titanic survival prediction.

This project focuses on:
- data preprocessing
- feature engineering
- machine learning pipelines
- model comparison
- evaluation metrics
- classification analysis

---

# Data Source

The Titanic dataset is loaded directly from Seaborn:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
```

No local dataset file is required to run this notebook.

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
Evaluation
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- XGBoost

---

# Machine Learning Concepts Covered

## Data Preprocessing

- Missing value handling
- OneHotEncoder
- StandardScaler
- ColumnTransformer
- Scikit-learn Pipelines
- Data leakage prevention

## Model Training

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

# Models Compared

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.843 |
| Random Forest | 0.810 | 0.787 | 0.696 | 0.738 | 0.840 |
| XGBoost | 0.804 | 0.758 | 0.724 | 0.741 | 0.809 |

---

# Final Model Evaluation

The Logistic Regression model was selected as the final baseline model because it is simple, interpretable, and achieved strong ROC-AUC performance.

Classification report summary:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0 - Did not survive | 0.81 | 0.89 | 0.85 | 110 |
| 1 - Survived | 0.79 | 0.67 | 0.72 | 69 |

Overall accuracy: **0.80**

The model performed better on class `0` than class `1`, meaning it was better at identifying passengers who did not survive than passengers who survived.

---

# Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

# Key Learning Outcomes

This project helped build understanding of:
- how preprocessing workflows are structured
- why feature engineering matters
- how Scikit-learn Pipelines prevent data leakage
- how to compare multiple classification models
- how to interpret classification metrics
- how to document a machine learning project professionally

---

# Repository Structure

```text
titanic-ml-pipeline/
│
├── images/
│   └── confusion_matrix.png
│
├── notebook/
│   └── titanic_pipeline.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation optimization
- Feature importance analysis
- ROC Curve visualization
- Modular Python scripts for training and evaluation
- Deployment with FastAPI

---

# Author

Melek Kuru
