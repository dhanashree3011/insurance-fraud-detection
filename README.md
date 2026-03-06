# 🛡️ InsureGuard — AI Insurance Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An end-to-end machine learning system that detects fraudulent auto insurance claims using a tuned Random Forest classifier — with a beautiful interactive dashboard built in Streamlit.

---

## Problem Statement

Insurance fraud costs the industry billions annually. The number of fraud cases detected is far lower than those actually committed. InsureGuard uses machine learning to analyze claim patterns, customer history, and incident characteristics to flag suspicious claims automatically.

---

##  Live Demo

**[Try the App](https://your-streamlit-app-url.streamlit.app)**

---

##  Model Performance

| Metric    | Score  |
|-----------|--------|
| AUC-ROC   | 0.929  |
| F1-Score  | 0.853  |
| Precision | 0.859  |
| Recall    | 0.848  |
| Accuracy  | 0.854  |

---

## Project Structure

```
insurance-fraud-detection/
│
├── app/
│   └── app.py                  ← Streamlit UI
│
├── notebooks/
│   ├── Auto_Insurance_EDA.ipynb       ← Exploratory Data Analysis
│   └── Model_Training_Tuning.ipynb    ← Model training notebook
│
├── src/
│   └── train.py                ← Full training pipeline script
│
├── models/
│   ├── best_model.pkl          ← Trained Random Forest model
│   ├── scaler.pkl              ← Feature scaler
│   ├── feature_names.json      ← Feature list
│   ├── label_encoders.pkl      ← Category encoders
│   └── model_report.json       ← Metrics summary
│
├── data/
│   └── raw/
│       └── insurance_claims.csv ← Kaggle dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/dhanashree3011/insurance-fraud-detection.git
cd insurance-fraud-detection

# 2. Create virtual environment
python -m venv fraud_env
fraud_env\Scripts\activate      # Windows
source fraud_env/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Get insurance_claims.csv from:
# https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
# Place it in: data/raw/insurance_claims.csv

# 5. Train the model
python src/train.py

# 6. Run the app
streamlit run app/app.py
```

---

## Features

### Dashboard
- Monthly fraud trend line chart
- Fraud vs legitimate donut chart
- Fraud rate by incident type, hour of day, claim amount

### Fraud Predictor
- Fill in 18 claim fields across 3 categories
- Instant fraud/legitimate prediction
- Animated risk gauge (0–100%)
- Detailed risk signal breakdown

### Analytics
- Upload any claims CSV
- Interactive distribution explorer
- Fraud rate by any categorical feature

### Model Performance
- All 5 metrics in visual cards
- Radar chart across metrics
- Best hyperparameters display
- Cross-validation F1 scores

---

## ML Pipeline

```
Raw Data
   ↓
Feature Engineering   (claim_premium_ratio, is_high_claim, is_night_incident...)
   ↓
NaN Imputation        (median for numeric, 0 for rest)
   ↓
Class Balancing       (SMOTE / manual oversampling — 12% → 50% fraud)
   ↓
Train/Test Split      (80/20 stratified)
   ↓
Train 3 Models        (Logistic Regression, Random Forest, Gradient Boosting)
   ↓
GridSearchCV Tuning   (5-fold CV on Random Forest)
   ↓
Cross Validation      (5-fold F1 = 0.871 ± 0.018)
   ↓
Save Artifacts        (model, scaler, encoders, feature list, report)
```

---

## Tech Stack

| Layer       | Technology                     |
|-------------|-------------------------------|
| Language    | Python 3.10+                  |
| ML          | Scikit-learn, Imbalanced-learn|
| UI          | Streamlit                     |
| Charts      | Plotly                        |
| Data        | Pandas, NumPy                 |
| Persistence | Joblib                        |
| Notebook    | Jupyter                       |

---

## Dataset

**Source:** [Auto Insurance Claims — Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)

- 15,420 claims · 40 features · 12% fraud rate
- Features: policy details, incident info, claim amounts, customer profile

---

## License

MIT License — feel free to use and modify.

---
