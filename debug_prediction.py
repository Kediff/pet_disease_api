# debug_prediction.py
import pickle
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

MODEL_PATH = "model/pethealth_bilstm_model.keras"
TOKENIZER_PATH = "model/tokenizer (2).pkl"
ENCODER_PATH = "model/label_encoder.pkl"
MAXLEN = 60

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Working dir:", os.getcwd())
print("Model exists?", os.path.exists(MODEL_PATH))
print("Tokenizer exists?", os.path.exists(TOKENIZER_PATH))
print("Encoder exists?", os.path.exists(ENCODER_PATH))

# Load
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

print("\nLabel encoder classes (index -> class):")
for idx, cls in enumerate(label_encoder.classes_):
    print(idx, "->", cls)

# The test string
text = "bad breath and bleeding gums"
cleaned = clean_text(text)
seq = tokenizer.texts_to_sequences([cleaned])
pad = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')

print("\nOriginal text:", text)
print("Cleaned text :", cleaned)
print("Token sequence:", seq)
# Show top word indices and their words
print("\nWord index lookups (token -> word):")
word_index = tokenizer.word_index
# invert a small mapping for readability
inv = {v:k for k,v in word_index.items()}
used_tokens = [t for t in seq[0] if t!=0]
print("Used token ids:", used_tokens)
print("Used words:", [inv.get(t, "<UNK>") for t in used_tokens])

# Predict
proba = model.predict(pad)
pred_index = np.argmax(proba, axis=1)[0]
topk = np.argsort(proba[0])[::-1][:5]  # top 5
print("\nPrediction probabilities (top classes):")
for i in topk:
    print(f"{i} ({label_encoder.classes_[i]}): {proba[0][i]:.4f}")
print("\nPred index:", pred_index, "->", label_encoder.inverse_transform([pred_index])[0])
