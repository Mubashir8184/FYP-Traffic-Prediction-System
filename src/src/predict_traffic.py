import pandas as pd
import joblib
from datetime import datetime

# ----------------------------
#  LOAD TRAINED MODEL
# ----------------------------
model_path = r"D:\Real Time Traffic Flow Prediction\models\rf_traffic_model.joblib"
model = joblib.load(model_path)

# ----------------------------
#  FUNCTION TO CREATE FEATURES
# ----------------------------
def create_features(input_datetime):
    dt = pd.to_datetime(input_datetime)

    features = {
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "weekday": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
    }

    return pd.DataFrame([features])

# ----------------------------
#  PREDICT FUNCTION
# ----------------------------
def predict_traffic(input_datetime):
    X = create_features(input_datetime)
    prediction = model.predict(X)[0]
    return prediction


# ----------------------------
#  RUN EXAMPLE PREDICTION
# ----------------------------
if __name__ == "__main__":
    user_time = input("Enter date & time (DD-MM-YYYY HH:MM): ")
    result = predict_traffic(user_time)
    print(f"\nPredicted Traffic Volume at {user_time}: {int(result)} vehicles")
