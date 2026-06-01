import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path

# =========================
# Page Config — MUST BE FIRST
# =========================

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# =========================
# Imports that trigger Streamlit
# (load_model after set_page_config)
# =========================

from tensorflow.keras.models import load_model

# =========================
# Path Setup
# =========================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "rnn_model.h5"

# =========================
# Load Model
# =========================

@st.cache_resource
def load_rnn_model():
    return load_model(MODEL_PATH)

model = load_rnn_model()

# =========================
# UI
# =========================

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image to predict which number it is (0–9)")

uploaded = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    # =========================
    # Open & Display Image
    # =========================

    img = Image.open(uploaded).convert("L")

    st.subheader("📷 Uploaded Image")
    st.image(img, width=200)

    # =========================
    # Preprocess
    # (RNN input: (1, 28, 28) — 28 timesteps of 28 features)
    # =========================

    img_array = np.array(img)

    # Invert if image is black-on-white (MNIST is white-on-black)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)

    img_resized = img.resize((28, 28))

    img_array = np.array(img_resized).astype("float32") / 255.0

    img_array = img_array.reshape(1, 28, 28)

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(img_array, verbose=0)

    predicted_digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction)) * 100

    # =========================
    # Results
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Predicted Digit: {predicted_digit}**")

    with col2:
        st.info(f"**Confidence: {confidence:.2f}%**")

    # =========================
    # Probability Chart
    # =========================

    st.subheader("📊 Prediction Probabilities")

    probabilities = prediction[0]

    prob_df = pd.DataFrame(
        {"Probability": probabilities},
        index=[f"Digit {i}" for i in range(10)]
    )

    st.bar_chart(prob_df)

    st.subheader("🔢 All Probabilities")

    for digit, prob in enumerate(probabilities):
        bar = "█" * int(prob * 30)
        label = "← Predicted" if digit == predicted_digit else ""
        st.write(
            f"Digit {digit}: {prob * 100:5.2f}%  {bar} {label}"
        )

# =========================
# Model Info
# =========================

with st.expander("ℹ️ Model Architecture"):

    st.write("**Model:** Recurrent Neural Network (LSTM)")
    st.write("**Dataset:** MNIST Handwritten Digits")
    st.write("**Input:** 28 timesteps × 28 features (one row per timestep)")

    st.code("""
LSTM(128, return_sequences=True)
Dropout(0.3)
LSTM(64)
Dropout(0.3)
Dense(64, relu)
Dense(10, softmax)
    """)

# =========================
# Footer
# =========================

st.caption("Built using Streamlit & TensorFlow/Keras")