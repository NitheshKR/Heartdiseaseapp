import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load model and preprocessing tools ---
model = load_model("heart_disease_model.h5")
scaler = joblib.load("scaler.joblib")
selected_features = joblib.load("selected_features.joblib")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health details below to predict the risk of heart disease.")

# --- Input Fields based on selected features ---
inputs = {}
for feature in selected_features:
    if feature in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'HvyAlcoholConsump', 'Sex']:
        inputs[feature] = st.selectbox(f"{feature}", [0, 1])
    elif feature == 'BMI':
        inputs[feature] = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    elif feature == 'Age':
        inputs[feature] = st.slider("Age", 18, 100, 40)
    else:
        inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Convert inputs to model format
input_data = np.array([[inputs[feature] for feature in selected_features]])
input_data = scaler.transform(input_data)

# --- Predict ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    result = (prediction > 0.5).astype(int)

    if result == 1:
        st.error("🚨 The model predicts a **High Risk of Heart Disease**.")
    else:
        st.success("✅ The model predicts a **Low Risk of Heart Disease**.")

st.caption("Model built with Deep Learning and deployed using Streamlit.")
