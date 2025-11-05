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
TOKENIZER_PATH = "model/tokenizer (1).pkl"
ENCODER_PATH = "model/label_encoder (1).pkl"

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
        load_resources()
        data = request.get_json() or {}
        symptoms = data.get("symptoms", "")

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Normalize symptoms (split by comma, and/or etc.)
        if isinstance(symptoms, list):
            items = [str(s).strip() for s in symptoms if str(s).strip()]
        else:
            s = str(symptoms).strip()
            s = re.sub(r"\band\b", ",", s)
            items = [p.strip() for p in re.split(r",|;|/|\||\n|\r", s) if p.strip()]

        if len(items) == 0:
            return jsonify({"error": "No valid symptoms parsed from input"}), 400

        processed_items = []
        for it in items:
            s_it = str(it).strip()
            if len(s_it.split()) == 1:
                s_it = f"My pet has {s_it}"
            processed_items.append(s_it)

        cleaned_items = [clean_text(it) for it in processed_items]
        seqs = tokenizer.texts_to_sequences(cleaned_items)
        logging.info("Parsed items: %s", items)
        logging.info("Cleaned items: %s", cleaned_items)
        logging.info("Tokenized sequences lengths: %s", [len(s) for s in seqs])

        if all(len(s) == 0 for s in seqs):
            msg = "Tokenizer produced empty sequences. Please check the input or tokenizer vocabulary."
            logging.error(msg)
            return jsonify({"error": msg}), 400

        non_empty_indices = [i for i, s in enumerate(seqs) if len(s) > 0]
        empty_indices = [i for i, s in enumerate(seqs) if len(s) == 0]
        results_by_index = {}

        for i in empty_indices:
            results_by_index[i] = {
                "symptom": items[i],
                "predictions": [],
                "error": "Input could not be tokenized by the model tokenizer",
            }

        if len(non_empty_indices) > 0:
            non_empty_seqs = [seqs[i] for i in non_empty_indices]
            pads = pad_sequences(non_empty_seqs, maxlen=MAXLEN, padding="post", truncating="post")
            pred_probas = model.predict(pads)

            threshold = 0.3
            for j, proba in enumerate(pred_probas):
                proba = np.asarray(proba)
                above_thresh_indices = np.where(proba >= threshold)[0]
                sorted_indices = above_thresh_indices[np.argsort(proba[above_thresh_indices])[::-1]]

                # Decode and store predictions
                predictions = [
                    {"disease": label_encoder.inverse_transform([i])[0], "confidence": float(proba[i])}
                    for i in sorted_indices
                ]

                # Always include top-1 even if all are below threshold
                if len(predictions) == 0:
                    top_index = int(np.argmax(proba))
                    predictions = [{
                        "disease": label_encoder.inverse_transform([top_index])[0],
                        "confidence": float(proba[top_index])
                    }]

                orig_index = non_empty_indices[j]
                entry = {
                    "symptom": items[orig_index],
                    "predictions": predictions,
                    "disease": predictions[0]["disease"],
                    "confidence": round(predictions[0]["confidence"], 3)
                }
                results_by_index[orig_index] = entry

        results = [results_by_index.get(i, {"symptom": items[i], "predictions": [], "error": "No prediction available"}) for i in range(len(items))]

        response = {"input": symptoms, "predictions": results}

        # Compatibility for single symptom input
        if len(results) == 1:
            response.update({
                "prediction": results[0].get("disease"),
                "confidence": results[0].get("confidence"),
            })

        return jsonify(response)

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
