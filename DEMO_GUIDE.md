# ML Lab — Presentation & Live Demonstration Guide

This guide is designed for presenting the **ML Lab** capstone project during live demonstrations, laboratory reviews, and academic viva evaluations. Follow this structured 5-to-10 minute presentation script to highlight the core machine learning methodology, interactive platform capabilities, and empirical discoveries.

---

## 1. Quick Setup Checklist Before the Presentation

1. **Activate the virtual environment & start the server:**
   ```bash
   source .venv/bin/activate
   python backend/main.py
   ```
2. **Open browser:** Navigate to `http://localhost:8000/`.
3. **Verify backend status:** Check the bottom-left sidebar indicator shows **`Backend Online (FastAPI Port 8000)`** with a green pulse dot.
4. **Display setup:** Maximize the browser window for optimal dashboard visibility (full desktop layout with persistent sidebar).

---

## 2. Walkthrough Script (5–10 Minutes)

### Act 1: The ML Command Center (`#/`) — 1 Minute
* **What to say:**
  > "Good morning/afternoon, evaluators. Welcome to **ML Lab**, our interactive machine learning platform built for the 23CSE301 Capstone. Rather than presenting static slides, our platform demonstrates 22 fully trained and evaluated models across three distinct machine learning disciplines: **Continuous Regression**, **Binary Classification**, and **Unsupervised Clustering**."
* **What to click & show:**
  - Point to the **3 Track Cards**:
    1. *Regression Lab* (UCI Air Quality, 10 models, Best: Random Forest $R^2 = 0.985$).
    2. *Classification Lab* (UCI Credit Approval, 10 models, Best: Tuned Random Forest $91.3\%$ Accuracy, $0.968$ ROC-AUC).
    3. *Clustering Lab* (UR3 Robotic Telemetry, K-Means & Agglomerative, $2.44\times$ Hazard Ratio).
  - Point to the **Summary Metric Cards** and the **Recent Experiments Feed** populated dynamically from our backend API.
  - Click on **"Enter Regression Lab →"** (or click *Regression Lab* in the sidebar).

---

### Act 2: Regression Lab — Pollution Prediction Simulator (`#/regression`) — 2 Minutes
* **What to say:**
  > "In the Regression track, we tackle continuous environmental monitoring on the UCI Air Quality benchmark. Our goal is to predict ground-truth benzene concentration `C6H6(GT)` in $\mu\text{g/m}^3$ from co-located chemical sensors. We trained and evaluated 10 distinct regression algorithms."
* **What to click & show:**
  1. **Interactive Prediction:**
     - Click the **"Rush Hour Peak"** preset button.
     - Note how carbon monoxide and nitrogen oxides automatically jump to high-traffic levels.
     - Click **"Predict C6H6(GT)"**. Show the resulting concentration value (e.g., $16.8\,\mu\text{g/m}^3$) and the **`High Pollution`** warning pill.
  2. **The What-If Simulator (Standout Feature):**
     - Click **"Compare Across All 10 Models (What-If)"**.
     - Point to the horizontal comparison bars:
       > *"Notice how the exact same sensor input is instantaneously evaluated across all 10 trained pipelines—from linear baselines (Ridge, Lasso) to non-linear ensembles (Random Forest, Gradient Boosting). Random Forest correctly captures the non-linear saturation effect of hydrocarbons."*
  3. **Diagnostics Tab:**
     - Click the **"3. Diagnostics"** tab.
     - Show the **Predicted vs. Actual scatter plot** clustering tightly along the dashed $y = x$ reference line.
     - Show the **Residual plot**: residuals are zero-mean centered with constant variance across the prediction range.
  4. **Feature Insights Tab:**
     - Click **"4. Feature Insights"**.
     - Point out that **`PT08.S2` (NMHC sensor) accounts for 48.6%** of predictive importance.
     - Highlight our explainability note: *Feature importance indicates observational correlation within the ensemble, not physical causality.*

---

### Act 3: Classification Lab — Model Face-Off (`#/classification`) — 3 Minutes
* **What to say:**
  > "Next is the Classification track using the UCI Credit Approval benchmark. The task is binary decision outcome (`+` Approved / `−` Denied). We implemented the complete 10-model portfolio spanning Review 1 baselines and Review 2 ensemble architectures."
* **What to click & show:**
  1. **Compliance & Disclaimer:**
     - Point to the banner:
       > *"Notice our visible educational disclaimer: this is strictly an academic ML demonstration, not calibrated for commercial lending decisions."*
  2. **Applicant Simulation:**
     - Click the **"High Approval Profile"** preset $\rightarrow$ Click **"Evaluate Credit Application"**.
     - Show the **`APPROVED`** badge, the probability gauge ($95.4\%$), and the top feature drivers.
     - Click the **"High Risk Profile"** preset $\rightarrow$ Click **"Evaluate Credit Application"**.
     - Show the **`REJECTED`** badge and risk score ($91.5\%$).
  3. **Feature Drivers Insight:**
     - Point out **`A9` (Prior Credit History / Verified Cards)** as the single dominant feature driver at **41.2%**:
       > *"In the training data, applicants with verified credit history (`A9 = 't'`) have a 78.7% approval rate, compared to just 7.0% for those without (`A9 = 'f'`)."*
  4. **Model Face-Off (Standout Feature):**
     - Click **"2. Compare (Model Face-Off)"** tab.
     - In *Challenger Model A*, choose **Random Forest (Tuned)**.
     - In *Challenger Model B*, choose **Logistic Regression** (or *Bagging* / *Gradient Boosting*).
     - Point to the side-by-side metric cards (Accuracy, ROC-AUC, Precision, Recall, F1) and the automated **Head-to-Head Test Split Delta** text box.
     - Scroll down to show the **All 10 Classification Algorithms table** containing all 10 evaluated models.
  5. **Evaluation Tab:**
     - Click **"3. Evaluation"** tab.
     - Show the **$2\times 2$ Confusion Matrix** ($TN=73, FP=4, FN=8, TP=53$) with $91.3\%$ accuracy and $94.8\%$ specificity.
     - Point to the continuous **ROC Curve** with an outstanding **$0.968$ AUC**.

---

### Act 4: Clustering Lab — Cluster Playground (`#/clustering`) — 2.5 Minutes
* **What to say:**
  > "The Clustering track is our most visual workspace. We analyzed 7,409 frames of UR3 industrial robotic telemetry across 19 sensor channels to discover operating regimes without supervision."
* **What to click & show:**
  1. **Zero-Leakage Protocol:**
     > *"A critical methodological requirement: the ground-truth safety label `Robot_ProtectiveStop` was strictly quarantined during preprocessing, scaling, and cluster fitting. It was used exclusively post-hoc."*
  2. **Interactive Playground:**
     - In the controls bar, move the **K slider** from $2$ to $3$, then click **"Refit Cluster Model"**.
     - Show how the Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics recompute in real time!
     - Move K back to **2** (our optimal configuration) and refit.
  3. **Elbow Method:**
     - Click **"2. Choose Cluster Count (Elbow)"** tab.
     - Show the within-cluster sum of squares inertia curve and the global peak in Silhouette Score ($0.3117$) at $k=2$.
  4. **Hierarchical Dendrogram:**
     - Click **"3. Dendrogram"** tab.
     - Show the distance cutoff threshold at $d=3.8$ isolating the two macro-clusters.
  5. **PCA vs. t-SNE Projections:**
     - Click **"4. Visualize (PCA & t-SNE)"** tab.
     - Toggle between **`PCA 2D`** (linear variance decomposition: $31.5\% + 16.5\% = 48.0\%$) and **`t-SNE 2D`** (non-linear manifold).
     - **Move your mouse over points on the canvas:** Show the interactive tooltip revealing sample ID, cluster assignment, and 2D coordinates in real time.
  6. **Post-Hoc Hazard Ratio (Key Discovery):**
     - Click **"5. Evaluate & Hazard Interpret"** tab.
     - Highlight the big headline metric: **`2.44× Relative Hazard Ratio`**:
       > *"Even though our model received zero safety labels during training, Cluster 0 exhibited a 4.76% emergency stop rate (187 stops) versus only 1.95% in Cluster 1 (39 stops). Unsupervised clustering autonomously identified the physical high-stress operating state susceptible to safety trips."*

---

### Act 5: Model Comparison Studio & Dataset Explorer (`#/compare`, `#/datasets`) — 1 Minute
* **What to click & show:**
  1. **Comparison Studio:**
     - Click **"Compare Models"** in the sidebar.
     - Click the filter pills: **Regression (10)**, **Classification (10)**, **Clustering (2)**, or **All Disciplines (22)**.
     - Point out the directionality guide (*Higher is better* vs *Lower is better*).
     - Click **"Export CSV"** $\rightarrow$ Instantly downloads `ML_Lab_Capstone_Leaderboard.csv`.
  2. **Dataset Explorer:**
     - Click **"Dataset Explorer"** in the sidebar.
     - Switch between **UCI Air Quality**, **UCI Credit Approval**, and **UR3 Telemetry**.
     - Show the schema definitions and the step-by-step preprocessing diagrams.
* **Closing Statement:**
  > "In summary, ML Lab bridges rigorous machine learning experimentation with an intuitive, production-grade interface. All 22 algorithms across all 3 tracks are fully functional, reproducible in our notebooks, and accessible via our REST API. Thank you, and we welcome your questions."

---

## 3. Anticipated Evaluator Questions & Winning Answers

### Q1: "Are these predictions real, or is the UI hardcoded?"
* **Answer:**
  > "Every prediction is 100% computed live through our FastAPI backend (`backend/main.py`). When you click 'Predict', a JSON payload containing the exact form features is sent to `/api/v1/classification/predict` or `/api/v1/regression/predict`, passed through our fitted scikit-learn pipelines (`models/all_classification_models.joblib`), and returned with live probabilities and feature importance scores. You can open Developer Tools (F12) $\rightarrow$ Network Tab to inspect the live HTTP responses."

### Q2: "Why does Random Forest outperform Linear Regression in the Air Quality task?"
* **Answer:**
  > "Chemical and meteorological sensors exhibit non-linear physical behaviors: atmospheric dispersion decreases exponentially with humidity, and sensor resistances saturate at high pollutant concentrations. Linear regression achieves an $R^2$ of $0.942$, but Random Forest captures these multi-sensor interactions and threshold effects without manual polynomial expansion, achieving an $R^2$ of $0.985$ and cutting RMSE from $1.84$ down to $0.98\,\mu\text{g/m}^3$."

### Q3: "How did you prevent data leakage in your preprocessing pipelines?"
* **Answer:**
  > "In both regression and classification, we performed an 80/20 train/test split *before* any imputation, encoding, or feature scaling. Transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`) were fit *strictly* on the training split, and the frozen transformers were applied to the test split. In clustering, ground-truth safety stop labels were completely isolated during fitting and evaluated strictly post-hoc."

### Q4: "Why did you choose $k=2$ for UR3 robot clustering instead of higher $k$?"
* **Answer:**
  > "While increasing $k$ naturally reduces within-cluster sum of squares (inertia), our elbow plot shows the sharpest decrease at $k=2$. Furthermore, the intrinsic **Silhouette Score** achieves its global peak at $k=2$ ($0.3117$), dropping steadily for $k \ge 3$. Cutting the hierarchical dendrogram at distance $d=3.8$ independently validates two primary structural macro-clusters."

### Q5: "What was the result of hyperparameter tuning in classification?"
* **Answer:**
  > "In Section A3 of `classification.ipynb`, we tuned our champion Random Forest estimator using `RandomizedSearchCV` across `n_estimators`, `max_depth`, `min_samples_split`, and `max_features`. Tuning produced measurable improvements across all metrics: test Accuracy rose from $87.68\%$ to **$91.30\%$**, weighted F1 rose from $0.8757$ to **$0.9124$**, and ROC-AUC rose from $0.9452$ to **$0.9681$**."
