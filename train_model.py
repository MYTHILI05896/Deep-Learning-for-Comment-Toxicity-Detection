import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# 1. Load dataset
# ==============================
print("Loading dataset...")
train_df = pd.read_csv("data/train.csv")

# Define label columns (multi-label classification)
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    train_df["comment_text"].values,
    train_df[label_cols].values,
    test_size=0.2,
    random_state=42
)

# ==============================
# 2. Text preprocessing
# ==============================
print("Preprocessing text...")
max_words = 20000   # vocabulary size
max_len = 150       # max sequence length

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# ==============================
# 3. Build Model
# ==============================
print("Building model...")
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPool1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(label_cols), activation="sigmoid")   # multi-label output
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 4. Training
# ==============================
print("Training model...")
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint("models/toxicity_model.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    batch_size=128,
    epochs=10,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==============================
# 5. Save tokenizer
# ==============================
import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Training completed. Model and tokenizer saved in 'models/' folder.")
