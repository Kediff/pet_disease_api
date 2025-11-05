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
MODEL_PATH = "model/pethealth_bilstm_model.keras"
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

        # Load model and other assets
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
        data = request.get_json() or {}
        symptoms = data.get("symptoms", "")
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Normalize input into a list of symptom strings (items).
        # Accept either a list of strings or a single string that may contain
        # multiple symptoms separated by common delimiters.
        if isinstance(symptoms, list):
            items = [str(s).strip() for s in symptoms if str(s).strip()]
        else:
            # Ensure we operate on a string for splitting and the one-word fix.
            s = str(symptoms).strip()
            # If the user provided a single word like "vomiting", add a short
            # context to help the tokenizer — but only for single-word strings.
            if len(s.split()) == 1:
                s = f"My pet has {s}"

            parts = re.split(r",|;|/|\||\n|\r|\band\b", s)
            items = [p.strip() for p in parts if p and p.strip()]

        if len(items) == 0:
            return jsonify({"error": "No valid symptoms parsed from input"}), 400

        # Tokenize and pad all items in one batch for efficiency
        cleaned_items = [clean_text(it) for it in items]
        seqs = tokenizer.texts_to_sequences(cleaned_items)
        # Debug logging to help diagnose empty-token problems
        logging.info("Parsed items: %s", items)
        logging.info("Cleaned items: %s", cleaned_items)
        logging.info("Tokenized sequences lengths: %s", [len(s) for s in seqs])

        # If tokenizer produced no tokens for all items, surface a helpful error
        if all(len(s) == 0 for s in seqs):
            msg = (
                "Tokenizer produced empty sequences for all inputs. "
                "This usually means the tokenizer vocabulary doesn't contain the input words "
                "or the cleaning removed all characters. Check tokenizer and input."
            )
            logging.error(msg)
            return jsonify({"error": msg}), 400
        pads = pad_sequences(seqs, maxlen=MAXLEN, padding="post", truncating="post")

        pred_probas = model.predict(pads)  # shape: (n_items, n_classes)
        logging.info("Model output shape: %s", np.atleast_2d(pred_probas).shape)
        try:
            logging.info("Label encoder classes count: %d", len(label_encoder.classes_))
        except Exception:
            logging.exception("Could not read label_encoder.classes_")

        results = []
        # If model outputs shape (n, ) because of single-class weirdness, normalize
        pred_probas = np.atleast_2d(pred_probas)

        # For each symptom, support multi-label outputs by thresholding.
        # If no class crosses the threshold, fall back to top-1.
        threshold = 0.3
        for i, proba in enumerate(pred_probas):
            proba = np.asarray(proba)
            # find indices over threshold
            if proba.ndim == 1 and proba.size > 1:
                indices = np.where(proba >= threshold)[0]
                if len(indices) == 0:
                    indices = [int(np.argmax(proba))]

                try:
                    labels = label_encoder.inverse_transform(indices)
                except Exception:
                    logging.exception("Label encoder transform failed")
                    labels = [str(idx) for idx in indices]

                confidences = [round(float(proba[idx]), 3) for idx in indices]
                preds = [
                    {"disease": lab, "confidence": conf}
                    for lab, conf in zip(labels, confidences)
                ]
            else:
                # fallback for unexpected shapes
                idx = int(np.argmax(proba))
                conf = float(np.max(proba))
                try:
                    lab = label_encoder.inverse_transform([idx])[0]
                except Exception:
                    logging.exception("Label encoder transform failed")
                    lab = str(idx)
                preds = [{"disease": lab, "confidence": round(conf, 3)}]

            entry = {"symptom": items[i], "predictions": preds}
            # keep single-item compatibility
            if len(preds) == 1:
                entry.update({"disease": preds[0]["disease"], "confidence": preds[0]["confidence"]})

            results.append(entry)

        # If only one item, keep compatibility by returning single-prediction fields
        response = {
            "input": symptoms,
            "predictions": results,
        }   
        if len(results) == 1:
            response.update({
                "prediction": results[0]["disease"],
                "confidence": results[0]["confidence"],
            })

        return jsonify(response)

        # end predict

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
