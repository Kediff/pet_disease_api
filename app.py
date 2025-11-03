from flask import Flask, request, jsonify
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import os
import logging

# NOTE: tensorflow is imported lazily inside load_resources() because importing
# it at module import time can use a lot of memory and cause some hosts to OOM
# or crash during startup. Delaying the import until the model is needed makes
# startup lighter and surfaces import/load errors in the request logs.

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/pethealth_bilstm_model.keras"
TOKENIZER_PATH = "model/tokenizer (2).pkl"
ENCODER_PATH = "model/label_encoder.pkl"

model = None
tokenizer = None
label_encoder = None
MAXLEN = 60


def load_resources():
    """Load model and supporting files only once."""
    global model, tokenizer, label_encoder
    if model is None:
        # Lazy import tensorflow to avoid heavy imports at module import time
        try:
            import tensorflow as tf
        except Exception as e:
            logging.exception("Failed to import tensorflow")
            raise

        # Validate model and artifact files exist early with clear messages
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Encoder file not found at {ENCODER_PATH}")

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            logging.exception("Error loading Keras model")
            raise

        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
        except Exception:
            logging.exception("Error loading tokenizer")
            raise

        try:
            with open(ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
        except Exception:
            logging.exception("Error loading label encoder")
            raise

        logging.info("✅ Model and tokenizer loaded.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/predict", methods=["POST"])
def predict_disease():
    try:
        logging.getLogger().setLevel(logging.INFO)
        load_resources()  # Load model only on first request
        data = request.get_json()
        symptoms_text = data.get("symptoms", "")
        if not symptoms_text:
            return jsonify({"error": "No symptoms provided"}), 400

        cleaned = clean_text(symptoms_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

        pred_proba = model.predict(pad)
        pred_index = np.argmax(pred_proba, axis=1)[0]
        confidence = float(np.max(pred_proba))
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        return jsonify({
            "input": symptoms_text,
            "prediction": pred_label,
            "confidence": round(confidence, 3),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "Pet Disease Prediction API is running 🚀"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
