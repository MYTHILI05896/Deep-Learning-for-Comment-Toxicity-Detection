import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# --- constants (must match training) ---
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 150

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/toxicity_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

st.title("🛡️ Multi-Label Toxic Comment Detection")
st.write("Enter a comment. The model predicts probabilities for each toxicity category.")

# Single prediction
text = st.text_area("💬 Your comment")
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(padded, verbose=0)[0]
        overall_toxic = any(p > 0.5 for p in probs)

        # Show per-label probabilities
        df = pd.DataFrame({
            "label": LABELS,
            "probability": probs,
            "flag (>0.5)": (probs > 0.5)
        })
        st.subheader("Per-label probabilities")
        st.dataframe(df)

        st.subheader("Overall")
        if overall_toxic:
            st.error("⚠️ Toxic (one or more labels > 0.5)")
        else:
            st.success("✅ Not Toxic (all labels ≤ 0.5)")

# Bulk predictions
st.subheader("📂 Bulk CSV Prediction")
uploaded = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
    if "comment_text" not in data.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        seqs = tokenizer.texts_to_sequences(data["comment_text"].astype(str))
        padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
        preds = model.predict(padded, verbose=0)
        out = pd.DataFrame(preds, columns=[f"prob_{l}" for l in LABELS])
        flags = (preds > 0.5).astype(int)
        out_flags = pd.DataFrame(flags, columns=[f"flag_{l}" for l in LABELS])
        result = pd.concat([data["comment_text"], out, out_flags], axis=1)
        st.dataframe(result.head(20))
        st.download_button(
            "⬇️ Download predictions",
            result.to_csv(index=False).encode(),
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )
