# <p align="center">🚦 BorderFlow MLflow  </p>
### <p align="center">End-to-End MLOps Experiment Tracking for Border Traffic Prediction</p>

<p align="center">
  <img src="https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow">
  <img src="https://img.shields.io/badge/Scikit--Learn-Regression-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/MLOps-Pipeline-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10+-brightgreen?style=for-the-badge&logo=python">
</p>

---

## 📌 Overview

**BorderFlow MLflow** is an end-to-end machine learning experiment tracking system built using MLflow to predict U.S. border crossing traffic volume.

Unlike toy datasets, this project uses real government transportation data, demonstrating a practical production-style MLOps workflow.

---

## 🚀 What This Project Demonstrates

- Multi-model training pipeline  
- MLflow experiment tracking  
- Model performance comparison  
- Automatic metric logging (RMSE, MAE, R²)  
- Model artifact versioning  
- Reproducible ML workflow  

---

## 🧠 Problem Statement

Predict the number of **border crossings (Value)** using:

- Border (US-Canada / US-Mexico)  
- Port Name  
- Measure (Vehicle Type)  
- Date (Year, Month, Day)  
- Weekend indicator  

This simulates real-world traffic forecasting and infrastructure planning.

---

## 🏗 Architecture

```
Border Dataset (CSV)
        ↓
Feature Engineering
        ↓
Multi-Model Training
        ↓
MLflow Experiment Tracking
        ↓
Metric Logging
        ↓
Model Artifact Logging
        ↓
Model Comparison Dashboard
```

---

## 📊 Models Trained

- Linear Regression  
- Ridge Regression  
- Support Vector Regressor  
- Random Forest Regressor  
- Decision Tree Regressor  

Each model is logged as a separate MLflow run.

---

## 📈 Metrics Tracked

For every model:

- RMSE  
- MAE  
- R² Score  

All metrics are automatically logged to the MLflow UI.

---

## 📂 Project Structure

```
borderflow-mlflow/
│
├── border.py                  # Main training + MLflow logging script
├── argv_exp.py                # CLI-based experiment file
├── requirements.txt
├── Border_Crossing_Entry_Data.csv
└── mlruns/                    # Auto-generated MLflow tracking folder
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/borderflow-mlflow.git
cd borderflow-mlflow
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python border.py
```

Start MLflow UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

You will see:

- Separate runs for each model  
- Logged metrics  
- Performance comparison dashboard  
- Stored model artifacts  

---

## 🔬 Feature Engineering

The pipeline automatically:

- Converts Date → datetime  
- Extracts Year, Month, Day  
- Creates Weekend flag  
- Encodes categorical features  

Ensuring structured and reproducible experiments.

---

## 🔁 Reproducibility

- Fixed random seed  
- Consistent train-test split  
- Logged experiments  
- Stored model artifacts  

Every experiment can be traced and reproduced.

---

## 🔮 Future Improvements

- Hyperparameter tuning  
- MLflow Model Registry (Staging → Production)  
- Docker containerization  
- CI/CD integration  
- Remote MLflow tracking server  

---

## 👨‍💻 Author

**Ankan Ghosh**  
Data Science & MLOps Enthusiast
