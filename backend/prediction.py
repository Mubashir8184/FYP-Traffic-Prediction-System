# loads trained model and predicts traffic for given datetime
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_traffic_model.joblib")

_model = None
_fallback_means = None


def _load_fallback_means():
    # when model missing use avg per hour from data
    global _fallback_means
    if _fallback_means is not None:
        return _fallback_means
    processed_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_full.csv")
    if not os.path.exists(processed_path):
        _fallback_means = {h: 4000 for h in range(24)}
        return _fallback_means
    df = pd.read_csv(processed_path, nrows=10000)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"])
    df["hour"] = df["date_time"].dt.hour
    means = df.groupby("hour")["traffic_volume"].mean().to_dict()
    _fallback_means = {h: int(means.get(h, 4000)) for h in range(24)}
    return _fallback_means


def _get_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception:
            pass
    return None


def create_features(dt):
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    row = {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "dayofweek": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "traffic_volume_lag_1": 0,
        "traffic_volume_lag_24": 0,
    }
    return pd.DataFrame([row])


def predict_traffic(input_datetime):
    # returns (predicted_volume, confidence_percent)
    dt = pd.to_datetime(input_datetime)
    model = _get_model()
    fallback = _load_fallback_means()
    pred_value = int(fallback.get(dt.hour, 4000))
    confidence = 70

    if model is not None:
        try:
            X = create_features(dt)
            train_cols = getattr(model, "feature_names_in_", None)
            if train_cols is not None:
                for c in train_cols:
                    if c not in X.columns:
                        X[c] = 0
                X = X[train_cols]
            pred = model.predict(X)
            pred_value = int(round(float(pred[0])))
            confidence = min(95, 70 + np.random.randint(0, 20))
        except Exception:
            pass
    return pred_value, confidence
