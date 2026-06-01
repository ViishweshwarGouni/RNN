import os
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

# =========================
# Create models folder
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# Load Dataset
# =========================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# =========================
# Preprocess
# (RNN treats each image as 28 timesteps of 28 features)
# =========================

X_train = X_train / 255.0   # shape: (60000, 28, 28)
X_test  = X_test  / 255.0   # shape: (10000, 28, 28)

# =========================
# Build RNN Model
# =========================

model = Sequential([

    LSTM(
        128,
        input_shape=(28, 28),
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')

])

# =========================
# Compile
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Train
# =========================

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# =========================
# Evaluate
# =========================

loss, acc = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"Test Loss     : {loss:.4f}")

# =========================
# Save Model
# =========================

model.save("models/rnn_model.h5")

print("\nModel saved to models/rnn_model.h5")
