from flask import Flask, request, jsonify
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)

# ======================================================
# 🔹 Model Paths
# ======================================================
MODEL_PATH = "model/pet_health_lstm_model_2.keras"
TOKENIZER_PATH = "model/tokenizer (2).pkl"
ENCODER_PATH = "model/label_encoder.pkl"

# ======================================================
# 🔹 Global Variables
# ======================================================
model = None
tokenizer = None
label_encoder = None
MAXLEN = 60

# ======================================================
# 🔹 Load resources only once
# ======================================================
def load_resources():
    """Load model and supporting files only once."""
    global model, tokenizer, label_encoder
    if model is None:
        try:
            import tensorflow as tf
        except Exception as e:
            logging.exception("Failed to import tensorflow")
            raise

        # Validate paths
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Encoder file not found at {ENCODER_PATH}")

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            logging.info("✅ Model, tokenizer, and encoder loaded successfully.")
        except Exception:
            logging.exception("Error loading model or tokenizer")
            raise

# ======================================================
# 🔹 Clean text
# ======================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================================================
# 🔹 Predict Route
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_disease():
    try:
        logging.getLogger().setLevel(logging.INFO)
        load_resources()  # Load model only on first request
        data = request.get_json()
        symptoms_text = data.get("symptoms", "")
        if not symptoms_text:
            return jsonify({"error": "No symptoms provided"}), 400

        # 🧠 Smart fix for one-word inputs
        if len(symptoms_text.split()) == 1:
            symptoms_text = f"My pet has {symptoms_text}"

        # Clean and tokenize
        cleaned = clean_text(symptoms_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

        # ======================================================
        # 🔹 New Improved Multi-Label Prediction Logic
        # ======================================================
        pred_proba = model.predict(pad)[0]

        # Get indices of all predictions above threshold (e.g., 0.3)
        threshold = 0.3
        above_thresh_indices = np.where(pred_proba >= threshold)[0]

        # Sort them by probability (descending)
        sorted_indices = above_thresh_indices[np.argsort(pred_proba[above_thresh_indices])[::-1]]

        # Decode their labels and probabilities
        predictions = [
            {"label": label_encoder.inverse_transform([i])[0], "confidence": float(pred_proba[i])}
            for i in sorted_indices
        ]

        # Always include the top prediction, even if all are below threshold
        if len(predictions) == 0:
            top_index = int(np.argmax(pred_proba))
            predictions = [{
                "label": label_encoder.inverse_transform([top_index])[0],
                "confidence": float(pred_proba[top_index])
            }]

        return jsonify({
            "input": symptoms_text,
            "predictions": predictions
        })

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

# ======================================================
# 🔹 Home Route
# ======================================================
@app.route("/")
def home():
    return jsonify({"message": "🐾 Pet Disease Prediction API is running 🚀"})

# ======================================================
# 🔹 Run Server
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

