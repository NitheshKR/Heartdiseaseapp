# app.py (ONNX + robust scaler loader)
import streamlit as st
import joblib, numpy as np, os, pandas as pd
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort

st.set_page_config(page_title="Heart Disease Risk (ONNX)", layout="centered")
st.title("Heart Disease Prediction (ONNX runtime)")

# ---------- Helpers ----------
def load_selected_features(path="selected_features.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        # fallback
        return ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes','PhysActivity','HvyAlcoholConsump','Sex','Age']

def load_scaler_safe(joblib_path="scaler.joblib", params_path="scaler_params.npz"):
    try:
        scaler = joblib.load(joblib_path)
        return scaler
    except Exception as e:
        try:
            d = np.load(params_path)
            mean = d['mean']
            scale = d['scale']
            scaler = StandardScaler()
            scaler.mean_ = np.array(mean)
            scaler.scale_ = np.array(scale)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            return scaler
        except Exception as e2:
            raise RuntimeError(f"Failed to load scaler: joblib error: {e}; params error: {e2}. "
                               "Upload scaler.joblib or scaler_params.npz to the app folder.")

def preprocess_input_dict(values_dict, selected_features, scaler):
    # build df
    row = pd.DataFrame([values_dict], columns=selected_features)
    # map common strings to numeric if present
    mapping = {"Yes":1, "No":0, "Male":1, "Female":0}
    for c in selected_features:
        if row[c].dtype == object:
            row[c] = row[c].map(mapping).fillna(row[c])
    X = row.values.astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled

# ---------- Load artifacts ----------
selected_features = load_selected_features("selected_features.joblib")
try:
    scaler = load_scaler_safe("scaler.joblib", "scaler_params.npz")
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# Load ONNX model
ONNX_PATH = "heart_model.onnx"
if not os.path.exists(ONNX_PATH):
    st.error(f"ONNX model '{ONNX_PATH}' not found. Upload heart_model.onnx to the app folder.")
    st.stop()
ort_sess = ort.InferenceSession(ONNX_PATH)
inp_name = ort_sess.get_inputs()[0].name

# ---------- UI ----------
st.sidebar.write("Model inputs:")
st.sidebar.write(selected_features)

st.subheader("Enter patient data")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    highbp = st.selectbox("HighBP", ["Yes","No"])
    highchol = st.selectbox("HighChol", ["Yes","No"])
    cholcheck = st.selectbox("CholCheck", ["Yes","No"])
with col2:
    bmi = st.number_input("BMI", 10.0, 60.0, value=25.0, format="%.2f")
    smoker = st.selectbox("Smoker", ["Yes","No"])
    stroke = st.selectbox("Stroke", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    phys = st.selectbox("PhysActivity", ["Yes","No"])
    hvyalc = st.selectbox("HvyAlcoholConsump", ["Yes","No"])

values = {
    "HighBP": 1 if highbp=="Yes" else 0,
    "HighChol": 1 if highchol=="Yes" else 0,
    "CholCheck": 1 if cholcheck=="Yes" else 0,
    "BMI": float(bmi),
    "Smoker": 1 if smoker=="Yes" else 0,
    "Stroke": 1 if stroke=="Yes" else 0,
    "Diabetes": 1 if diabetes=="Yes" else 0,
    "PhysActivity": 1 if phys=="Yes" else 0,
    "HvyAlcoholConsump": 1 if hvyalc=="Yes" else 0,
    "Sex": 1 if sex=="Male" else 0,
    "Age": int(age)
}

st.write("Inputs:")
st.json(values)

if st.button("Predict"):
    X_scaled = preprocess_input_dict(values, selected_features, scaler)  # shape (1,11)
    # Predict with ONNX
    preds = ort_sess.run(None, {inp_name: X_scaled.astype(np.float32)})[0]  # shape (1,1)
    prob = float(np.ravel(preds)[0])
    st.metric("Predicted risk (%)", f"{prob*100:.2f}%")
    st.success("Risk level: " + ("High" if prob>0.7 else "Medium" if prob>0.4 else "Low"))
    st.caption("Disclaimer: Demo only — not medical advice.")

