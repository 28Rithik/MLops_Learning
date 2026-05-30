from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

FEATURE_COLUMNS = [
    "speed_kmph",
    "accel_x",
    "accel_y",
    "brake_pressure",
    "steering_angle",
    "throttle",
    "lane_deviation",
    "phone_usage",
    "headway_distance",
    "reaction_time",
]

MODEL_FILES = {
    "Random Forest": ARTIFACTS / "random_forest.joblib",
    "XGBoost": ARTIFACTS / "xgboost.joblib",
}

SAFETY_MESSAGES = {
    "Safe": "Driver behavior looks normal. Continue monitoring.",
    "Distracted": "Attention risk detected. Avoid phone use and refocus on the road.",
    "Aggressive": "Aggressive driving detected. Reduce speed and smooth steering/braking.",
}

st.set_page_config(
    page_title="Driver Behavior Monitor",
    page_icon="🚗",
    layout="wide",
)

st.title("Driver Behavior & Vehicle Safety Monitoring System")
st.caption("Choose a model, enter one record or upload a CSV, and get behavior predictions.")

@st.cache_resource
def load_label_encoder():
    path = ARTIFACTS / "label_encoder.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing label encoder: {path}")
    return joblib.load(path)

@st.cache_resource
def load_model(model_name: str):
    path = MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)

def ensure_artifacts_exist():
    missing = []
    if not ARTIFACTS.exists():
        missing.append(str(ARTIFACTS))
    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            missing.append(str(model_path))
    if not (ARTIFACTS / "label_encoder.joblib").exists():
        missing.append(str(ARTIFACTS / "label_encoder.joblib"))
    return missing

def make_input_frame(values: dict) -> pd.DataFrame:
    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return frame

def decode_prediction(prediction, label_encoder):
    value = np.asarray(prediction)[0]
    if isinstance(value, (str, np.str_)):
        return str(value)
    return label_encoder.inverse_transform(np.asarray(prediction).astype(int))[0]


def predict_with_model(model, label_encoder, input_frame: pd.DataFrame):
    prediction = model.predict(input_frame)
    predicted_label = decode_prediction(prediction, label_encoder)

    confidence = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_frame)[0]
        probabilities = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, proba)
        }
        confidence = float(np.max(proba))

    return predicted_label, confidence, probabilities

def render_result(predicted_label: str, confidence: float | None):
    left, right = st.columns(2)
    with left:
        st.metric("Predicted Behavior", predicted_label)
    with right:
        if confidence is None:
            st.metric("Confidence", "Unavailable")
        else:
            st.metric("Confidence", f"{confidence:.2%}")

    st.info(SAFETY_MESSAGES.get(predicted_label, "Prediction generated."))

def render_probability_chart(probabilities: dict[str, float] | None):
    if not probabilities:
        return
    proba_df = pd.DataFrame(
        [{"Behavior": label, "Probability": prob} for label, prob in probabilities.items()]
    ).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Behavior"))

def prediction_form(defaults: dict[str, float], key_prefix: str):
    inputs = {}
    cols = st.columns(2)

    field_configs = [
        ("speed_kmph", "Speed (kmph)", 0.0, 160.0, 0.1),
        ("accel_x", "Acceleration X", -10.0, 10.0, 0.01),
        ("accel_y", "Acceleration Y", -10.0, 10.0, 0.01),
        ("brake_pressure", "Brake Pressure", 0.0, 120.0, 0.1),
        ("steering_angle", "Steering Angle", -180.0, 180.0, 0.1),
        ("throttle", "Throttle", 0.0, 100.0, 0.1),
        ("lane_deviation", "Lane Deviation", 0.0, 5.0, 0.01),
        ("phone_usage", "Phone Usage (0 or 1)", 0.0, 1.0, 1.0),
        ("headway_distance", "Headway Distance", 0.0, 100.0, 0.1),
        ("reaction_time", "Reaction Time", 0.0, 5.0, 0.01),
    ]

    for index, (col_name, label, min_val, max_val, step) in enumerate(field_configs):
        with cols[index % 2]:
            inputs[col_name] = st.number_input(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(defaults.get(col_name, 0.0)),
                step=float(step),
                key=f"{key_prefix}_{col_name}",
            )
    return inputs

missing = ensure_artifacts_exist()
if missing:
    st.error("Missing required files:")
    for item in missing:
        st.write(item)
    st.stop()

label_encoder = load_label_encoder()

model_name = st.radio(
    "Choose model",
    list(MODEL_FILES.keys()),
    horizontal=True,
    index=0,
)

model = load_model(model_name)

tab_single, tab_upload = st.tabs(["Single Prediction", "CSV Upload"])

with tab_single:
    st.subheader("Manual single prediction")
    st.write("Enter one driver record to predict the behavior class.")

    default_values = {feature: 0.0 for feature in FEATURE_COLUMNS}
    default_values["speed_kmph"] = 40.0
    default_values["throttle"] = 30.0
    default_values["headway_distance"] = 20.0
    default_values["reaction_time"] = 1.0

    with st.form("single_prediction_form"):
        values = prediction_form(default_values, "single")
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_frame = make_input_frame(values)
        predicted_label, confidence, probabilities = predict_with_model(
            model, label_encoder, input_frame
        )
        render_result(predicted_label, confidence)
        render_probability_chart(probabilities)

        st.write("Input used for prediction")
        st.dataframe(input_frame, use_container_width=True)

with tab_upload:
    st.subheader("CSV batch prediction")
    st.write("Upload a CSV with the same feature columns used by the model.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
        if missing_cols:
            st.error("Missing required columns:")
            for col in missing_cols:
                st.write(col)
            st.stop()

        extra_cols = [col for col in batch_df.columns if col not in FEATURE_COLUMNS]
        if extra_cols:
            st.warning("Extra columns will be ignored:")
            st.write(extra_cols)

        predict_button = st.button("Run batch prediction")

        if predict_button:
            input_batch = batch_df[FEATURE_COLUMNS].copy()
            predictions = model.predict(input_batch)
            predicted_labels = [
    str(p) if isinstance(p, (str, np.str_))
    else label_encoder.inverse_transform(np.asarray([p]).astype(int))[0]
    for p in predictions
]

            results_df = batch_df.copy()
            results_df["predicted_label"] = predicted_labels

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_batch)
                confidence = probabilities.max(axis=1)
                results_df["prediction_confidence"] = confidence

            st.success(f"Predicted {len(results_df)} rows.")

            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="driver_behavior_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Prediction summary")
            summary = results_df["predicted_label"].value_counts().reset_index()
            summary.columns = ["Behavior", "Count"]
            st.bar_chart(summary.set_index("Behavior"))