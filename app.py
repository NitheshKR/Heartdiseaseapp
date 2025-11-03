# app.py
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# optional imports (onnxruntime or tf)
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

from sklearn.preprocessing import StandardScaler

# ---------- Config ----------
SELECTED_FEATURES_PATH = "selected_features.joblib"
SCALER_JOBLIB_PATH = "scaler.joblib"
SCALER_PARAMS_PATH = "scaler_params.npz"
CSV_PATH = "heart_disease_health_indicators.csv"  # used only if scaler needs rebuilding
ONNX_PATH = "heart_model.onnx"
HDF5_PATH = "heart_disease_model.h5"
# expected number of features (11)
EXPECTED_FEATURE_LEN = 11

# default selected features (fallback)
DEFAULT_SELECTED = [
    'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
    'Diabetes','PhysActivity','HvyAlcoholConsump','Sex','Age'
]
# ----------------------------

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("Heart Disease Risk Predictor")

def load_selected_features(path=SELECTED_FEATURES_PATH):
    if os.path.exists(path):
        try:
            sel = joblib.load(path)
            if isinstance(sel, (list, tuple, np.ndarray)):
                return list(sel)
        except Exception:
            pass
    return DEFAULT_SELECTED

def save_scaler_params(scaler, params_path=SCALER_PARAMS_PATH):
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        np.savez(params_path, mean=scaler.mean_, scale=scaler.scale_)

def rebuild_scaler_from_csv(csv_path=CSV_PATH, selected=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file for rebuilding scaler not found at '{csv_path}'. Upload it to the app folder or provide scaler_params.npz.")
    df = pd.read_csv(csv_path)
    # ensure selected features exist
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns for scaler rebuild: {missing}")
    X = df[selected].copy()
    # convert textual categories to numeric if necessary
    if X['Sex'].dtype == object:
        X['Sex'] = X['Sex'].map({'Male':1, 'Female':0}).fillna(0).astype(int)
    for c in ['HighBP','HighChol','CholCheck','Smoker','Stroke','Diabetes','PhysActivity','HvyAlcoholConsump']:
        if X[c].dtype == object:
            X[c] = X[c].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    scaler = StandardScaler().fit(X.values.astype(float))
    # save portable params
    save_scaler_params(scaler)
    return scaler

def load_scaler(joblib_path=SCALER_JOBLIB_PATH, params_path=SCALER_PARAMS_PATH, selected=None):
    # Try joblib
    if os.path.exists(joblib_path):
        try:
            scaler = joblib.load(joblib_path)
            if hasattr(scaler, "mean_") and len(scaler.mean_) == EXPECTED_FEATURE_LEN:
                return scaler
        except Exception:
            pass
    # Try params file
    if os.path.exists(params_path):
        try:
            d = np.load(params_path)
            mean = d['mean']
            scale = d['scale']
            if len(mean) == EXPECTED_FEATURE_LEN:
                scaler = StandardScaler()
                scaler.mean_ = mean
                scaler.scale_ = scale
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = len(mean)
                return scaler
        except Exception:
            pass
    # If we reach here, attempt rebuild from CSV if available
    if os.path.exists(CSV_PATH):
        try:
            scaler = rebuild_scaler_from_csv(csv_path=CSV_PATH, selected=selected)
            return scaler
        except Exception as e:
            raise RuntimeError(f"Failed to rebuild scaler from CSV: {e}")
    raise RuntimeError("No usable scaler found. Upload 'scaler.joblib' or 'scaler_params.npz', or upload the dataset CSV to rebuild the scaler.")

def fix_zero_scales(scaler):
    if not hasattr(scaler, "scale_"):
        return scaler
    scale = np.array(scaler.scale_)
    zeros = scale == 0
    if np.any(zeros):
        scale[zeros] = 1e-6
        scaler.scale_ = scale
        scaler.var_ = scaler.scale_ ** 2
    return scaler

def load_model_prefer(onnx_path=ONNX_PATH, h5_path=HDF5_PATH):
    # prefer ONNX for lighter runtime
    if os.path.exists(onnx_path) and ort is not None:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        return ("onnx", sess, input_name)
    # try HDF5 keras model
    if os.path.exists(h5_path) and load_model is not None:
        model = load_model(h5_path)
        return ("keras", model, None)
    # nothing found
    return (None, None, None)

# Load selected features
selected_features = load_selected_features()
if len(selected_features) != EXPECTED_FEATURE_LEN:
    # if stored selected_features lists dummified columns etc., fallback to default
    selected_features = DEFAULT_SELECTED

# Load or rebuild scaler
try:
    scaler = load_scaler(selected=selected_features)
    scaler = fix_zero_scales(scaler)
except Exception as e:
    st.error(f"Scaler loading error: {e}")
    st.stop()

# Load model
model_kind, model_obj, model_input_name = load_model_prefer()
if model_kind is None:
    st.error("No model found in app folder. Upload 'heart_model.onnx' (preferred) or 'heart_disease_model.h5'.")
    st.stop()

# UI inputs
st.subheader("Enter patient information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    highbp = st.selectbox("HighBP (High blood pressure?)", ["Yes","No"])
    highchol = st.selectbox("HighChol (High cholesterol?)", ["Yes","No"])
    cholcheck = st.selectbox("CholCheck (Cholesterol checked?)", ["Yes","No"])
with col2:
    bmi = st.number_input("BMI", 10.0, 60.0, value=25.26, format="%.2f")
    smoker = st.selectbox("Smoker", ["Yes","No"])
    stroke = st.selectbox("Stroke (history)?", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    phys = st.selectbox("PhysActivity (Physically active?)", ["Yes","No"])
    hvyalc = st.selectbox("HvyAlcoholConsump (Heavy alcohol?)", ["Yes","No"])

# Build input dict in expected order
values = {
    "HighBP": 1 if highbp == "Yes" else 0,
    "HighChol": 1 if highchol == "Yes" else 0,
    "CholCheck": 1 if cholcheck == "Yes" else 0,
    "BMI": float(bmi),
    "Smoker": 1 if smoker == "Yes" else 0,
    "Stroke": 1 if stroke == "Yes" else 0,
    "Diabetes": 1 if diabetes == "Yes" else 0,
    "PhysActivity": 1 if phys == "Yes" else 0,
    "HvyAlcoholConsump": 1 if hvyalc == "Yes" else 0,
    "Sex": 1 if sex == "Male" else 0,
    "Age": int(age)
}

if st.button("Predict"):
    # prepare row
    row = pd.DataFrame([values], columns=selected_features)
    # ensure mapping & numeric
    mapping = {"Yes":1, "No":0, "Male":1, "Female":0}
    for c in selected_features:
        if row[c].dtype == object:
            row[c] = row[c].map(mapping).fillna(row[c])
    try:
        X_raw = row.values.astype(float)
    except Exception as e:
        st.error(f"Input conversion error: {e}")
        st.stop()
    # scale
    try:
        X_scaled = scaler.transform(X_raw)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}. You may need to upload a correct 'scaler_params.npz' or the dataset CSV so the app can rebuild the scaler.")
        st.stop()

    # model predict
    try:
        if model_kind == "onnx":
            pred = model_obj.run(None, {model_input_name: X_scaled.astype(np.float32)})[0]
        else:
            pred = model_obj.predict(X_scaled)
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        st.stop()

    # parse probability
    try:
        prob = float(np.ravel(pred)[0])
    except Exception as e:
        st.error(f"Could not parse model output: {e}")
        st.stop()

    if np.isnan(prob) or np.isinf(prob):
        st.error("Model returned an invalid probability (NaN or Inf). Consider rebuilding scaler from Colab and uploading scaler_params.npz.")
        st.stop()

    pct = prob * 100.0
    st.metric("Predicted heart disease risk (%)", f"{pct:.2f}%")
    level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    st.success(f"Risk level: {level}")
    st.caption("Disclaimer: This is a demo tool for educational purposes; not medical advice.")
