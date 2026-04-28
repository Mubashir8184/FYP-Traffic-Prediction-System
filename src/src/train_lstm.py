"""
train_lstm.py

Train an LSTM model for Real-Time Traffic Flow Prediction.
Requires tensorflow and scikit-learn.

Run:
    pip install tensorflow
    python src/src/train_lstm.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Attempt to import tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    print("TensorFlow not found. Please run 'pip install tensorflow' first.")
    tf = None

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(DATA_DIR, "processed_full.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_traffic_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scalers.joblib")

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm():
    if tf is None:
        return

    print("Loading data...")
    if not os.path.exists(PROCESSED_FILE):
        print(f"Error: {PROCESSED_FILE} not found.")
        return

    df = pd.read_csv(PROCESSED_FILE)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)
    df = df.dropna()

    # Features and Target
    target_col = 'traffic_volume'
    feature_cols = [c for c in df.columns if c != target_col and c != 'date_time']
    
    # Filter numeric features
    df_numeric = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = df_numeric.columns.tolist()
    
    print(f"Using features: {feature_cols}")

    # Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_x = scaler_x.fit_transform(df[feature_cols])
    scaled_y = scaler_y.fit_transform(df[[target_col]])

    # Sequence Generation (e.g., 24 hours lookback)
    SEQ_LENGTH = 24
    print(f"Creating sequences with window size {SEQ_LENGTH}...")
    X, y = create_sequences(scaled_x, scaled_y, SEQ_LENGTH)

    # Split (Chronological)
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Train shape: {X_train.shape}")
    
    # Build Model
    print("Building LSTM model...")
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # 10 epochs for demonstration, can be increased
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Save
    model.save(MODEL_PATH)
    joblib.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y, 'feature_cols': feature_cols, 'seq_length': SEQ_LENGTH}, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scalers saved to {SCALER_PATH}")

    # Evaluation
    print("\nEvaluation on Test Data:")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"LSTM Test -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_lstm()
