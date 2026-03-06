from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_features, load_data


def train_model(data_path: str, model_output: str, metrics_output: str) -> None:
    df = load_data(data_path)
    df = build_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    split_date = df["date"].quantile(0.8)
    train_idx = df["date"] <= split_date

    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]

    numeric = [c for c in FEATURE_COLUMNS if c != "crypto_name"]
    categorical = ["crypto_name"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_date": str(split_date.date()),
        "target": "next_day_volatility",
    }

    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output)
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/crypto_data.csv")
    parser.add_argument("--model_out", default="models/volatility_model.joblib")
    parser.add_argument("--metrics_out", default="models/metrics.json")
    args = parser.parse_args()

    train_model(args.data, args.model_out, args.metrics_out)
