"""
train_models.py

Train a baseline model for Real-Time Traffic Flow Prediction.
Uses chronological train/validation/test split (no shuffling) and evaluates MAE & RMSE.

Run:
    python src/train_models.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# -------------------------
# Paths (absolute from project root)
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(processed_dir, "processed_full.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_traffic_model.joblib")

print("Looking for processed file at:", PROCESSED_FILE)

# -------------------------
# 1️⃣ Load processed data
# -------------------------
if not os.path.exists(PROCESSED_FILE):
    raise FileNotFoundError(f"Processed file not found at {PROCESSED_FILE}. Please run data_prep.py first.")

df = pd.read_csv(PROCESSED_FILE)

# Ensure date_time is datetime type and sorted
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

# -------------------------
# 2️⃣ Chronological Train/Val/Test Split (index-based)
# -------------------------
n = len(df)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

# -------------------------
# 3️⃣ Features and Target
# -------------------------
target = 'traffic_volume'
feature_cols = [c for c in df.columns if c != target and c != 'date_time']

X_train, y_train = train[feature_cols], train[target]
X_val, y_val     = val[feature_cols], val[target]
X_test, y_test   = test[feature_cols], test[target]

# Keep only numeric features
X_train = X_train.select_dtypes(include=[np.number])
X_val   = X_val.select_dtypes(include=[np.number])
X_test  = X_test.select_dtypes(include=[np.number])

print("X_train shape (numeric only):", X_train.shape)

# -------------------------
# 4️⃣ Train Baseline Model
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"Saved Random Forest model to {MODEL_PATH}")

# -------------------------
# 5️⃣ Evaluate Model
# -------------------------
def evaluate_model(model, X, y, dataset_name="Dataset"):
    if X.shape[0] == 0:
        print(f"{dataset_name} is empty, skipping evaluation.")
        return None, None
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"{dataset_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

print("\nEvaluation:")
evaluate_model(model, X_train, y_train, "Train")
evaluate_model(model, X_val, y_val, "Validation")
evaluate_model(model, X_test, y_test, "Test")
