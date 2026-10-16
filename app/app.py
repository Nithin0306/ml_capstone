"""
23CSE301 Machine Learning Capstone — Streamlit Deployment Alternative
File: app/app.py
Enables 1-click deployment on Streamlit Community Cloud (https://share.streamlit.io)
Loads pre-trained pipelines directly from models/ directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ML Capstone Intelligence Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0c0d11; color: #ffffff; }
    .stMetric { background: rgba(255, 255, 255, 0.04); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); }
    .metric-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# Resolve Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    class_art = joblib.load(os.path.join(MODELS_DIR, "classification_rf.joblib"))
    clust_art = joblib.load(os.path.join(MODELS_DIR, "clustering_kmeans.joblib"))
    reg_art   = joblib.load(os.path.join(MODELS_DIR, "regression_rf.joblib"))
    with open(os.path.join(MODELS_DIR, "leaderboard.json")) as f:
        lboard = json.load(f)
    return class_art, clust_art, reg_art, lboard

class_art, clust_art, reg_art, leaderboard = load_models()

# Sidebar Navigation
st.sidebar.title("🤖 ML Capstone Hub")
st.sidebar.caption("23CSE301 Review 2 Deployment")
tab_selection = st.sidebar.radio(
    "Select Track / View:",
    ["📊 Executive Overview", "💳 Credit Approval (Random Forest)", "🤖 UR3 Robot Diagnostics (K-Means)", "💨 Air Quality (Regression)", "🏆 Benchmark Leaderboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Team 3 Deliverables:**")
st.sidebar.markdown("• **Random Forest Classifier** (ROC-AUC: 0.962)")
st.sidebar.markdown("• **K-Means Clustering** (k=2, Sil: 0.312)")
st.sidebar.markdown("• **Air Quality Regressor** (R²: 0.999)")

# View 1: Overview
if tab_selection == "📊 Executive Overview":
    st.title("Autonomous Telemetry & Predictive Inference")
    st.markdown("End-to-end production pipelines for Credit Approval Classification, UR3 Robotic Diagnostics, and Air Quality Regression.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC (Random Forest)", "96.2%", "+11.7% vs Tree")
    col2.metric("Silhouette Score (K-Means)", "0.312", "k=2 Optimal")
    col3.metric("R² Score (Air Quality)", "0.999", "Random Forest")
    col4.metric("Benchmarked Models", "22", "Review 1 + 2")

    st.markdown("---")
    st.subheader("Leaderboard Preview")
    df_lb = pd.DataFrame(leaderboard)
    st.dataframe(df_lb, use_container_width=True)

# View 2: Classification
elif tab_selection == "💳 Credit Approval (Random Forest)":
    st.title("Credit Approval Predictor")
    st.caption("Tuned Random Forest Classifier (Review 2 Model) • Dataset: crx.data")

    preset = st.radio("Load Preset:", ["Custom Inputs", "🌟 Prime Applicant (Approve)", "⚠️ High-Risk Applicant (Reject)"], horizontal=True)

    default_age = 35.0 if "Prime" in preset else (22.0 if "High-Risk" in preset else 32.5)
    default_debt = 2.5 if "Prime" in preset else (9.5 if "High-Risk" in preset else 3.5)
    default_employed_yrs = 6.0 if "Prime" in preset else (0.5 if "High-Risk" in preset else 4.0)
    default_lines = 8 if "Prime" in preset else (0 if "High-Risk" in preset else 3)
    default_a9 = "t" if "Prime" in preset else ("f" if "High-Risk" in preset else "t")
    default_a10 = "t" if "Prime" in preset else ("f" if "High-Risk" in preset else "t")
    default_income = 6500.0 if "Prime" in preset else (120.0 if "High-Risk" in preset else 2500.0)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Applicant Age (A2)", value=float(default_age))
        debt = st.number_input("Debt Ratio (A3)", value=float(default_debt))
        employed_yrs = st.number_input("Years Employed (A8)", value=float(default_employed_yrs))
        lines = st.number_input("Credit Score / Active Lines (A11)", value=int(default_lines))

    with col2:
        a9 = st.selectbox("Credit History / Verification (A9)", ["t (Verified History)", "f (No History / Default)"], index=0 if default_a9 == "t" else 1)
        a10 = st.selectbox("Currently Employed (A10)", ["t (Employed)", "f (Unemployed)"], index=0 if default_a10 == "t" else 1)
        income = st.number_input("Annual Income (A15)", value=float(default_income))
        a14 = st.number_input("Postal Code / Numeric (A14)", value=160.0)

    if st.button("Compute Approval Probability", type="primary"):
        num_vals = [age, debt, employed_yrs, lines, a14, income]
        cat_vals = ['b', 'u', 'g', 'c', 'v', a9[0], a10[0], 'f', 'g']

        num_imp = class_art["num_imputer"].transform([num_vals])
        num_scl = class_art["scaler"].transform(num_imp)
        cat_imp = class_art["cat_imputer"].transform([cat_vals])
        cat_ohe = class_art["ohe"].transform(cat_imp)

        X_in = np.hstack([num_scl, cat_ohe])
        prob = class_art["model"].predict_proba(X_in)[0][1]
        approved = prob >= 0.5

        if approved:
            st.success(f"### Decision: APPROVED (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"### Decision: REJECTED (Risk Factor: {(1-prob)*100:.1f}%)")

        st.progress(float(prob))

        st.subheader("Top Driving Features (Gini Importance)")
        top_drivers = sorted(zip(class_art["feature_names"], class_art["importances"]), key=lambda x: x[1], reverse=True)[:5]
        df_imp = pd.DataFrame(top_drivers, columns=["Feature", "Importance Weight"])
        st.bar_chart(df_imp.set_index("Feature"))

# View 3: Clustering
elif tab_selection == "🤖 UR3 Robot Diagnostics (K-Means)":
    st.title("UR3 Robotic Telemetry Diagnostics")
    st.caption("Unsupervised K-Means ($k=2$) & 2D PCA Anomaly Projection • Dataset: ur3.xlsx")

    preset_clust = st.radio("Telemetry Mode:", ["Custom Readings", "🟢 Baseline Routine Motion", "🔴 High-Torque Stress Cycle"], horizontal=True)

    default_j0 = 2.8 if "Stress" in preset_clust else -0.4
    default_j1 = -4.2 if "Stress" in preset_clust else -1.1
    default_j2 = 3.6 if "Stress" in preset_clust else 0.6
    default_speed = 1.45 if "Stress" in preset_clust else 0.12
    default_temp = 64.0 if "Stress" in preset_clust else 38.5

    col1, col2 = st.columns(2)
    with col1:
        curr_j0 = st.slider("Base Current (Current_J0, A)", -5.0, 5.0, float(default_j0))
        curr_j1 = st.slider("Shoulder Current (Current_J1, A)", -5.0, 5.0, float(default_j1))
        curr_j2 = st.slider("Elbow Current (Current_J2, A)", -5.0, 5.0, float(default_j2))
        tool_curr = st.slider("Tool Current (A)", 0.0, 3.0, 0.2)
        speed_j0 = st.slider("Base Angular Velocity (Speed_J0, rad/s)", 0.0, 3.0, float(default_speed))
        temp_t0 = st.slider("Joint Temperature (°C)", 20.0, 90.0, float(default_temp))

    # Inference
    sensor_cols = clust_art["sensor_cols"]
    input_dict = {
        "Current_J0": curr_j0, "Current_J1": curr_j1, "Current_J2": curr_j2,
        "Tool_current": tool_curr, "Speed_J0": speed_j0, "Temperature_T0": temp_t0
    }
    vals = [input_dict.get(c, np.nan) for c in sensor_cols]
    vals_arr = np.array([vals])
    vals_imp = clust_art["imputer"].transform(vals_arr)
    vals_scl = clust_art["scaler"].transform(vals_imp)

    cluster_id = int(clust_art["model"].predict(vals_scl)[0])
    pca_coords = clust_art["pca"].transform(vals_scl)[0]

    with col2:
        st.subheader("Real-Time Diagnostic Result")
        if cluster_id == 0:
            st.warning("⚠️ **Cluster 0: High Electrical Load / Hazard Risk**")
            st.metric("Protective Stop Rate", "4.76%", "2.44× elevated incident risk")
        else:
            st.success("✅ **Cluster 1: Baseline Routine Motion**")
            st.metric("Protective Stop Rate", "1.95%", "Normal safe operation")

        st.caption(f"2D PCA Coordinates: PC1={pca_coords[0]:.2f}, PC2={pca_coords[1]:.2f}")

        # Plot 2D PCA Space
        bg_pts = np.array(clust_art["pca_background"])
        bg_labs = np.array(clust_art["labels_background"])
        df_pca = pd.DataFrame({"PC1": bg_pts[:, 0], "PC2": bg_pts[:, 1], "Cluster": [f"Cluster {l}" for l in bg_labs]})
        st.scatter_chart(df_pca, x="PC1", y="PC2", color="Cluster")

# View 4: Regression
elif tab_selection == "💨 Air Quality (Regression)":
    st.title("Air Quality Benzene Concentration Estimator")
    st.caption("Random Forest Regressor • Dataset: AirQualityUCI.csv")

    col1, col2 = st.columns(2)
    with col1:
        co = st.number_input("PT08.S1 (CO Sensor)", value=1360.0)
        nmhc = st.number_input("PT08.S2 (NMHC Sensor)", value=1046.0)
        nox = st.number_input("PT08.S3 (NOx Sensor)", value=1056.0)
    with col2:
        temp = st.number_input("Temperature (°C)", value=13.6)
        rh = st.number_input("Relative Humidity (%)", value=48.9)
        ah = st.number_input("Absolute Humidity", value=0.758)

    features = reg_art["features"]
    in_dict = {"PT08.S1(CO)": co, "PT08.S2(NMHC)": nmhc, "PT08.S3(NOx)": nox, "T": temp, "RH": rh, "AH": ah}
    reg_vals = [in_dict.get(c, np.nan) for c in features]
    reg_imp = reg_art["imputer"].transform([reg_vals])
    reg_scl = reg_art["scaler"].transform(reg_imp)

    pred_c6h6 = float(reg_art["model"].predict(reg_scl)[0])

    st.subheader(f"Estimated Benzene Concentration: **{pred_c6h6:.2f} μg/m³**")
    if pred_c6h6 < 5.0:
        st.success("Air Quality: Good (< 5 μg/m³)")
    elif pred_c6h6 < 10.0:
        st.warning("Air Quality: Moderate (5 - 10 μg/m³)")
    else:
        st.error("Air Quality: High Pollution (> 10 μg/m³)")

# View 5: Leaderboard
elif tab_selection == "🏆 Benchmark Leaderboard":
    st.title("Comprehensive Model Benchmark Leaderboard")
    st.caption("Consolidated metrics across Review 1 & Review 2 (22 algorithms)")
    st.dataframe(pd.DataFrame(leaderboard), use_container_width=True)
