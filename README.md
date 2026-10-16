# ML Lab — Interactive Machine Learning Capstone Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI%200.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.4+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/course-23CSE301%20ML%20Capstone-crimson.svg)]()

> **23CSE301 Machine Learning Capstone Project**  
> *Amrita School of Computing, Amrita Vishwa Vidyapeetham*  
> A unified experimentation, inference, and diagnostic studio spanning three machine learning disciplines: **Continuous Regression**, **Binary Classification**, and **Unsupervised Clustering**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [How to Run the Application](#how-to-run-the-application)
- [Frontend UI Workspaces](#frontend-ui-workspaces)
- [Machine Learning Tracks Summary](#machine-learning-tracks-summary)
- [REST API Reference](#rest-api-reference)
- [Jupyter Notebooks Overview](#jupyter-notebooks-overview)
- [License & Academic Disclaimer](#license--academic-disclaimer)

---

## Project Overview

**ML Lab** is an interactive, responsive analytics platform built to demonstrate real machine learning model inference, diagnostic validation, and empirical decision support:

1. **Continuous Regression (UCI Air Quality):** Predicts ground-truth benzene concentration `C6H6(GT)` ($\mu\text{g/m}^3$) across **10 regressors** with an interactive What-If simulator and residual diagnostics.
2. **Binary Classification (UCI Credit Approval):** Benchmarks **10 classifiers** (Review 1 baselines + Review 2 ensembles and neural architectures) with side-by-side Model Face-Offs, probability estimation, and interactive confusion matrices.
3. **Unsupervised Clustering (UR3 Robotic Telemetry):** Discovers operational regimes across 19 sensor signals using **K-Means** and **Agglomerative Hierarchical Clustering**, verified via PCA/t-SNE projections and a post-hoc $2.44\times$ safety hazard ratio discovery.

---

## Repository Structure

```text
ml_capstone_project/
├── backend/
│   ├── main.py                     # Production FastAPI server & REST API
│   └── server.py                   # Lightweight standalone HTTP server
├── frontend/
│   ├── index.html                  # Single-page application shell (6 views)
│   ├── style.css                   # Responsive dark charcoal/#BF0C4F theme
│   └── app.js                      # Client router, Canvas visualizers & API client
├── models/
│   ├── all_classification_models.joblib  # 10 fitted classifiers & preprocessor
│   ├── all_regression_models.joblib      # 10 fitted regressors & scalers
│   ├── clustering_kmeans.joblib          # K-Means, PCA/t-SNE & cluster metrics
│   └── leaderboard.json                  # Benchmarks across all 22 algorithms
├── notebooks/
│   ├── regression.ipynb            # Air Quality continuous regression pipeline
│   ├── classification.ipynb        # Complete 10-algorithm credit approval pipeline
│   └── clustering.ipynb            # UR3 telemetry clustering & safety validation
├── data/
│   ├── regression/                 # UCI Air Quality dataset (AirQualityUCI.csv)
│   ├── classification/             # UCI Credit Approval (crx.data / crx.names)
│   └── clustering/                 # UR3 industrial telemetry (ur3.xlsx)
├── requirements.txt                # Python environment dependencies
├── README.md                       # Main repository and execution documentation
└── DEMO_GUIDE.md                   # Step-by-step evaluator presentation guide
```

---

## How to Run the Application

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Linux, macOS, or Windows)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
# Navigate to project directory
cd ml_capstone_project

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Server
Start the unified FastAPI server which runs the ML inference engine and mounts the web application:

```bash
python backend/main.py
```

*Or run with Uvicorn directly:*
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the Platform
Open your browser and navigate to:
```
http://localhost:8000/
```

- **Frontend Application:** `http://localhost:8000/`
- **Interactive Swagger API Docs:** `http://localhost:8000/docs`
- **Alternative ReDoc API Docs:** `http://localhost:8000/redoc`

---

## Frontend UI Workspaces

The platform features 6 dedicated workspaces accessible via client-side hash routing:

| Route | Workspace | Key Highlights |
|:---|:---|:---|
| `#/` | **Overview / ML Command Center** | High-level metrics, 3 discipline cards, recent experiments activity feed, 4-step quick start. |
| `#/regression` | **Regression Lab** | 10 regressors, What-If multi-model simulator, sensor sliders, predicted-vs-actual scatter & residuals. |
| `#/classification` | **Classification Lab** | 10 classifiers, applicant profiles, Model Face-Off head-to-head comparison, confusion matrix & ROC curve. |
| `#/clustering` | **Clustering Lab** | Live K refitting (2–8), Elbow inertia plot, Dendrogram ($d=3.8$ cut), PCA & t-SNE 2D toggles, $2.44\times$ hazard ratio. |
| `#/compare` | **Model Comparison Studio** | Unified leaderboard across all 22 algorithms, filter pills, metric directionality badges, CSV export. |
| `#/datasets` | **Dataset Explorer** | Column schemas, missing value strategies, and pipeline diagrams for all 3 datasets. |

---

## Machine Learning Tracks Summary

### 1. Regression Track (UCI Air Quality)
- **Target:** `C6H6(GT)` (Ground-Truth Benzene concentration in $\mu\text{g/m}^3$)
- **Data:** 9,357 records, 13 chemical sensor channels. -200 sentinel values imputed with training median; zero test data leakage.
- **Top Performer:** **Random Forest Regressor** ($R^2 = 0.985$, $RMSE = 0.982\,\mu\text{g/m}^3$, $MAE = 0.512\,\mu\text{g/m}^3$).
- **Key Insight:** Sensor $PT08.S2$ (Non-Metanic Hydrocarbons) drives $48.6\%$ of predictive variance.

### 2. Classification Track (UCI Credit Approval)
- **Target:** Binary Credit Outcome ($+$ Approved / $-$ Denied)
- **Data:** 690 records, 15 demographic and financial attributes. Evaluated on 80/20 stratified split (138 test cases).
- **Consolidated 10-Algorithm Ranking:**
  1. **Tuned Random Forest:** $91.30\%$ Accuracy, $0.9681$ ROC-AUC, $0.9124$ F1
  2. **Bagging (Decision Tree):** $89.86\%$ Accuracy, $0.9640$ ROC-AUC, $0.8986$ F1
  3. **Logistic Regression:** $89.13\%$ Accuracy, $0.9600$ ROC-AUC, $0.8904$ F1
  4. **Decision Tree Classifier:** $88.41\%$ Accuracy, $0.9286$ ROC-AUC, $0.8844$ F1
  5. **Gradient Boosting:** $88.41\%$ Accuracy, $0.9508$ ROC-AUC, $0.8842$ F1
  6. **AdaBoost Classifier:** $87.68\%$ Accuracy, $0.9517$ ROC-AUC, $0.8761$ F1
  7. **Support Vector Machine (RBF):** $86.96\%$ Accuracy, $0.9478$ ROC-AUC, $0.8686$ F1
  8. **K-Nearest Neighbours ($k=7$):** $86.23\%$ Accuracy, $0.9499$ ROC-AUC, $0.8601$ F1
  9. **MLP Classifier:** $85.51\%$ Accuracy, $0.9212$ ROC-AUC, $0.8536$ F1
  10. **Gaussian Naive Bayes:** $78.99\%$ Accuracy, $0.9004$ ROC-AUC, $0.7774$ F1
- **Key Insight:** Attribute $A9$ (Prior Credit Verification / Active Cards) accounts for $41.2\%$ of tree split gain ($78.7\%$ approval for 't' vs $7.0\%$ for 'f').

### 3. Clustering Track (UR3 Robot Telemetry)
- **Target:** Unsupervised operating regime discovery. Ground-truth label `Robot_ProtectiveStop` strictly excluded during training.
- **Data:** 7,409 high-frequency time-series frames across 19 joint current, velocity, and temperature channels.
- **Partitioning vs Hierarchy:**
  - **K-Means ($k=2$):** Silhouette $0.3117$, Davies-Bouldin $1.3939$, Calinski-Harabasz $1997.1$.
  - **Agglomerative (Ward Linkage, $k=2$):** Silhouette $0.2132$, Davies-Bouldin $1.574$, Calinski-Harabasz $1109.3$.
- **Post-Hoc Safety Validation:**
  - Cluster 0: **4.76%** emergency stop rate ($187$ stops in $3,929$ cycles)
  - Cluster 1: **1.95%** emergency stop rate ($39$ stops in $1,998$ cycles)
  - **Hazard Ratio: 2.44×** (Proves autonomous detection of hazardous mechanical/electrical stress states without label leakage).

---

## REST API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/v1/health` | Service health status and backend metadata |
| `GET` | `/api/v1/datasets` | Metadata and schemas for all 3 capstone datasets |
| `GET` | `/api/v1/models/{track}` | Models available for `regression`, `classification`, or `clustering` |
| `GET` | `/api/v1/results/{track}` | Benchmark scores for all models in a track |
| `POST` | `/api/v1/regression/predict` | Predicts $C6H6(GT)$ for given sensor readings using selected model |
| `POST` | `/api/v1/regression/compare` | Runs What-If simulation across all 10 regressors on identical input |
| `GET` | `/api/v1/regression/diagnostics` | Returns test split predicted vs actual and residual distribution |
| `POST` | `/api/v1/classification/predict` | Predicts credit approval, probability, risk score, and top 5 drivers |
| `POST` | `/api/v1/classification/compare` | Predicts probability across all 10 classifiers on identical input |
| `GET` | `/api/v1/classification/evaluation/{model_id}` | Confusion matrix and evaluation metrics for selected classifier |
| `GET` | `/api/v1/clustering/config` | Clustering parameters, feature list, and cluster profiles |
| `POST` | `/api/v1/clustering/projection` | Returns 2D coordinates for `pca` or `tsne` projections |
| `POST` | `/api/v1/clustering/elbow` | Inertia values and silhouette scores across $k=2 \dots 10$ |
| `GET` | `/api/v1/clustering/dendrogram` | Linkage tree nodes and distance cutoff threshold |
| `POST` | `/api/v1/clustering/run` | Dynamically refits K-Means or Agglomerative for specified $k$ and linkage |
| `GET` | `/api/v1/experiments` | Experiment history and session validation logs |

---

## Jupyter Notebooks Overview

All three notebooks in `notebooks/` are self-contained, fully executed, and sequentially numbered:

1. [`notebooks/regression.ipynb`](notebooks/regression.ipynb) — EDA, -200 sentinel handling, train-fitted scaling, 10 regressors, residual plots, feature importances.
2. [`notebooks/classification.ipynb`](notebooks/classification.ipynb) — Complete 10-algorithm evaluation, Review 1 baselines, Review 2 ensembles, consolidated Section A2 comparison table, and Section A3 hyperparameter tuning.
3. [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) — 19-sensor data audit, PCA & t-SNE projections, K-Means elbow plot, Agglomerative dendrogram, Silhouette/DB/CH benchmarks, and post-hoc ground-truth hazard ratio validation.

---

## License & Academic Disclaimer

This project is created for academic coursework in **23CSE301 Machine Learning** at Amrita Vishwa Vidyapeetham. The Credit Approval predictor is an educational machine learning demonstration and is strictly not intended for real-world financial or credit decision-making.