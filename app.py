"""Streamlit demo: enter machine readings, get a failure-probability
prediction, a cost-aware recommendation, and a per-prediction SHAP
explanation showing which readings actually drove the score."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import engineer_features

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "results" / "models"
COST_OPTIMAL_THRESHOLD = 0.12

st.set_page_config(page_title="Predictive Maintenance Demo", page_icon="🔧")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer


model, scaler, explainer = load_artifacts()

st.title("Predictive Maintenance — Failure Risk Demo")
st.caption(
    "Live demo for [Predictive-Maintenance-Failure-Classification]"
    "(https://github.com/saijignas/Predictive-Maintenance-Failure-Classification). "
    "Random forest trained on the AI4I 2020 benchmark dataset — a synthetic-but-realistic "
    "dataset, not real factory sensor logs. Enter machine readings to see the model's "
    "failure probability, the recommendation at each threshold, and which readings drove it."
)

col1, col2 = st.columns(2)
with col1:
    machine_type = st.selectbox("Product quality variant", ["L", "M", "H"], index=0)
    air_temp = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
with col2:
    rotational_speed = st.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    tool_wear = st.slider("Tool wear (min)", 0, 260, 100, 1)

raw = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Type": machine_type,
    "Machine failure": 0,  # unused by engineer_features beyond building y
}])
X, _ = engineer_features(raw)
for col in ["type_L", "type_M", "type_H"]:
    if col not in X.columns:
        X[col] = False
X = X[scaler.feature_names_in_]

X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
proba = model.predict_proba(X_scaled)[0, 1]

st.subheader(f"Predicted failure probability: {proba:.1%}")

c1, c2 = st.columns(2)
with c1:
    flag_default = proba >= 0.5
    st.metric("Default threshold (0.50)", "FLAG" if flag_default else "OK")
with c2:
    flag_optimal = proba >= COST_OPTIMAL_THRESHOLD
    st.metric(f"Cost-optimal threshold ({COST_OPTIMAL_THRESHOLD})", "FLAG" if flag_optimal else "OK")

if flag_optimal and not flag_default:
    st.info(
        "The default 0.50 threshold would miss this one. The cost-optimal threshold "
        "(chosen because a missed failure costs far more than a false alarm — see README) "
        "flags it instead."
    )

st.subheader("Why the model predicted this")
shap_values = explainer(X_scaled)
if shap_values.values.ndim == 3:
    shap_values = shap_values[..., 1]

contrib = pd.Series(shap_values.values[0], index=X.columns).sort_values(key=abs, ascending=True)
st.bar_chart(contrib)
st.caption(
    "SHAP contribution of each reading to this specific prediction. Positive = pushes "
    "toward failure, negative = pushes toward normal."
)

st.divider()
st.caption(
    "This is a demo of the modeling methodology, not a maintenance decision tool for a "
    "real machine — the underlying dataset is synthetic and the cost assumptions behind "
    "the threshold are illustrative. Full write-up and code in the linked repo."
)
