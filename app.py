# app.py - Debug-friendly & robust Streamlit app for Heart Disease prediction
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Try to import optional libs
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

from sklearn.preprocessing import StandardScaler

# ----------------- Config -----------------
DEBUG = True   # Set False to hide detailed debug outputs in the UI
SELECTED_FEATURES_PATH = "selected_features.joblib"
SCALER_JOBLIB_PATH = "scaler.joblib"
SCALER_PARAMS_PATH = "scaler_params.npz"
ONNX_PATH = "heart_model.onnx"
HDF5_PATH = "heart_disease_model.h5"  # fallback
# ------------------------------------------

st.set_page_config(page_title="Heart Disease Risk (Debug)", layout="centered")
st.title("Heart Disease Prediction — Debug Mode" if DEBUG else "Heart Disease Prediction")

# ----------------- Helpers -----------------
def load_selected_features(path=SELECTED_FEATURES_PATH):
    try:
        selected = joblib.load(path)
        if not isinstance(selected, (list, tuple, np.ndarray)):
            raise ValueError("selected_features loaded but not a list")
        return list(selected)
    except Exception as e:
        st.warning(f"Could not load selected features from '{path}': {e}. Using default 11 features.")
        return ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes','PhysActivity','HvyAlcoholConsump','Sex','Age']

def load_scaler_safe(joblib_path=SCALER_JOBLIB_PATH, params_path=SCALER_PARAMS_PATH):
    # Try joblib first
    try:
        scaler = joblib.load(joblib_path)
        if DEBUG: st.write(f"Loaded scaler from {joblib_path} (type: {type(scaler)})")
        return scaler, "joblib"
    except Exception as e_job:
        if DEBUG: st.write(f"joblib.load failed: {e_job}")
        # fallback to params file
        try:
            npz = np.load(params_path)
            mean = npz['mean']
            scale = npz['scale']
            # reconstruct StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(mean)
            scaler.scale_ = np.array(scale)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            if DEBUG: st.write(f"Reconstructed StandardScaler from {params_path}")
            return scaler, "params"
        except Exception as e_params:
            # final fallback: return None and error
            raise RuntimeError(f"Failed to load scaler from joblib ({e_job}) and params ({e_params}). Upload scaler.joblib or scaler_params.npz.")

def fix_zero_scales(scaler):
    """
    Replace zero scales with a small epsilon to avoid division by zero during transform.
    Updates scaler in-place.
    """
    if not hasattr(scaler, 'scale_'):
        return scaler, False
    scale = np.array(getattr(scaler, 'scale_', None))
    if scale is None:
        return scaler, False
    zeros = (scale == 0)
    if np.any(zeros):
        eps = 1e-6
        scale_fixed = np.where(zeros, eps, scale)
        scaler.scale_ = scale_fixed
        scaler.var_ = scaler.scale_ ** 2
        # n_features_in_ should already be set
        return scaler, True
    return scaler, False

def load_model_auto(onnx_path=ONNX_PATH, h5_path=HDF5_PATH):
    # Try ONNX
    if os.path.exists(onnx_path) and ort is not None:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        return ("onnx", sess, input_name)
    # Else try HDF5 / Keras model
    if os.path.exists(h5_path) and load_model is not None:
        model = load_model(h5_path)
        return ("keras", model, None)
    # Nothing found
    return (None, None, None)

def preprocess_input_dict(values_dict, selected_features, scaler):
    # build df with correct column order
    row = pd.DataFrame([values_dict], columns=selected_features)
    # try to map common string categories to numeric
    mapping = {"Yes":1, "No":0, "Male":1, "Female":0}
    for c in selected_features:
        # if dtype/object try mapping
        if row[c].dtype == object or isinstance(row.loc[0, c], str):
            row[c] = row[c].map(mapping).fillna(row[c])
    # convert to numeric
    try:
        X_raw = row.values.astype(float)
    except Exception as e:
        raise ValueError(f"Could not convert input row to floats: {e}\nRow was: {row.to_dict(orient='records')}")
    # scale
    X_scaled = scaler.transform(X_raw)
    return X_raw, X_scaled

# ----------------- Load artifacts -----------------
selected_features = load_selected_features()
st.sidebar.write("Model input features (order):")
st.sidebar.write(selected_features)

try:
    scaler, scaler_src = load_scaler_safe()
except Exception as e:
    st.error(str(e))
    st.stop()

# fix zero scales if any
scaler, fixed = fix_zero_scales(scaler)
if fixed and DEBUG:
    st.warning("Detected zero scale values in scaler; replaced with small epsilon to avoid divide-by-zero.")

# load model (ONNX preferred)
model_kind, model_obj, model_input_name = load_model_auto()
if model_kind is None:
    st.error("No model found in app folder. Please upload 'heart_model.onnx' (preferred) or 'heart_disease_model.h5'.")
    st.stop()
else:
    st.sidebar.write(f"Loaded model type: {model_kind}")

# ----------------- UI Inputs -----------------
st.subheader("Enter patient data")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male","Female"])
    highbp = st.selectbox("HighBP", ["Yes","No"])
    highchol = st.selectbox("HighChol", ["Yes","No"])
    cholcheck = st.selectbox("CholCheck", ["Yes","No"])
with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.26, format="%.2f")
    smoker = st.selectbox("Smoker", ["Yes","No"])
    stroke = st.selectbox("Stroke", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    phys = st.selectbox("PhysActivity", ["Yes","No"])
    hvyalc = st.selectbox("HvyAlcoholConsump", ["Yes","No"])

# build the ordered dict
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

st.markdown("### Inputs (ordered)")
st.json(values)

# ----------------- Predict & Debug -----------------
if st.button("Predict"):
    # Preprocess & debug
    try:
        X_raw, X_scaled = preprocess_input_dict(values, selected_features, scaler)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        if DEBUG:
            st.exception(e)
        st.stop()

    # Debug prints
    if DEBUG:
        st.write("X_raw (before scaling):")
        st.write(X_raw)
        st.write("scaler.mean_ (length):", getattr(scaler, 'mean_', None).shape if hasattr(scaler, 'mean_') else None)
        st.write("scaler.mean_:", getattr(scaler, 'mean_', None))
        st.write("scaler.scale_ (length):", getattr(scaler, 'scale_', None).shape if hasattr(scaler, 'scale_') else None)
        st.write("scaler.scale_:", getattr(scaler, 'scale_', None))
        st.write("X_scaled (after scaling):")
        st.write(X_scaled)
        # check for NaN/Inf
        if np.any(np.isnan(X_raw)) or np.any(np.isinf(X_raw)):
            st.error("ERROR: X_raw contains NaN or Inf.")
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            st.error("ERROR: X_scaled contains NaN or Inf. Check scaler parameters or feature order.")
            st.stop()

    # Run prediction
    try:
        if model_kind == "onnx":
            # ONNX expects float32
            inp = {model_input_name: X_scaled.astype(np.float32)}
            preds = model_obj.run(None, inp)[0]
        else:
            preds = model_obj.predict(X_scaled)
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        if DEBUG:
            st.exception(e)
        st.stop()

    # Show raw preds and handle NaN
    if DEBUG:
        st.write("Raw model output (preds):")
        st.write(preds)

    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        st.error("Model produced NaN or Inf predictions. See debug outputs above.")
        st.stop()

    # flatten to scalar prob
    try:
        prob = float(np.ravel(preds)[0])
    except Exception as e:
        st.error(f"Could not parse model output to scalar: {e}")
        if DEBUG:
            st.exception(e)
        st.stop()

    if np.isnan(prob):
        st.error("Prediction is NaN. See debug outputs above.")
        st.stop()

    st.metric("Predicted risk (%)", f"{prob*100:.2f}%")
    st.success("Risk level: " + ("High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"))
    st.caption("Disclaimer: Demo only — not medical advice.")

    st.metric("Predicted risk (%)", f"{prob*100:.2f}%")
    st.success("Risk level: " + ("High" if prob>0.7 else "Medium" if prob>0.4 else "Low"))
    st.caption("Disclaimer: Demo only — not medical advice.")

