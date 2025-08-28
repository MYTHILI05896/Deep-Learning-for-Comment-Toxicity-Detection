import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from src.preprocessing import clean_text
from src.utils import preprocess_input

# Load model & tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("app/model.h5")
    with open("app/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

st.title("🛡️ Toxic Comment Detection (Deep Learning + Streamlit)")
st.write("Enter a comment and check if it's toxic.")

user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    X_pad = preprocess_input([cleaned], tokenizer)
    prediction = model.predict(X_pad)[0][0]
    label = "Toxic 😡" if prediction > 0.5 else "Not Toxic 🙂"
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {prediction:.2f}")

# Demo with test.csv
st.subheader("📊 Evaluate on Test Dataset")
if st.button("Run Evaluation"):
    test = pd.read_csv("data/test.csv")
    test['cleaned'] = test['comment_text'].apply(clean_text)
    X_pad = preprocess_input(test['cleaned'], tokenizer)
    test['prediction'] = (model.predict(X_pad) > 0.5).astype("int")
    st.dataframe(test[['comment_text', 'prediction', 'toxic']].head(20))
