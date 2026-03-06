from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS


def predict_one(model_path: str, payload: dict) -> float:
    model = joblib.load(model_path)

    # Keep only expected columns and preserve ordering.
    sample = pd.DataFrame([{col: payload[col] for col in FEATURE_COLUMNS}])
    prediction = float(model.predict(sample)[0])
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/volatility_model.joblib")
    parser.add_argument("--payload", required=True, help="JSON string with feature values")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    pred = predict_one(args.model, payload)
    print({"predicted_next_day_volatility": pred})
