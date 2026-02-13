import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title('🏥 Hospital Readmission Risk Predictor')

model = joblib.load('xgb_readmission_model.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('optimal_threshold.pkl')

st.sidebar.header('Patient Features')
num_procedures = st.sidebar.slider('Number of Lab Procedures', 0, 200, 50)
time_in_hospital = st.sidebar.slider('Days in Hospital', 0, 14, 5)
num_medications = st.sidebar.slider('Number of Medications', 0, 50, 15)
num_diagnoses = st.sidebar.slider('Number of Diagnoses', 0, 16, 8)

# Create prediction
features = np.array([[num_procedures, time_in_hospital, num_medications, num_diagnoses]])
prob = model.predict_proba(features)[0][1]
risk_level = "🔴 HIGH RISK" if prob >= threshold else "🟢 LOW RISK"

st.metric("Readmission Risk", f"{prob*100:.1f}%")
st.write(f"**Risk Classification:** {risk_level}")

if prob >= threshold:
    st.warning("⚠️ Patient flagged for care coordination!")
