from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["crypto_name", "date"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["current_volatility"] = (data["high"] - data["low"]) / (data["close"].abs() + EPS)
    data["daily_return"] = (data["close"] - data["open"]) / (data["open"].abs() + EPS)
    data["price_range_pct"] = (data["high"] - data["low"]) / (data["open"].abs() + EPS)
    data["candle_body_pct"] = (data["close"] - data["open"]).abs() / (data["open"].abs() + EPS)
    data["volume_to_marketcap"] = data["volume"] / (data["marketCap"].abs() + EPS)
    data["log_volume"] = np.log1p(np.clip(data["volume"], a_min=0, a_max=None))
    data["log_marketcap"] = np.log1p(np.clip(data["marketCap"], a_min=0, a_max=None))

    for col in ["close", "volume", "marketCap"]:
        data[f"{col}_lag1"] = data.groupby("crypto_name")[col].shift(1)
        data[f"{col}_pct_change_1d"] = data.groupby("crypto_name")[col].pct_change()

    for window in [3, 7]:
        data[f"close_ma_{window}"] = data.groupby("crypto_name")["close"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        data[f"volatility_ma_{window}"] = data.groupby("crypto_name")["current_volatility"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    # Predict next day's volatility using current-day market information.
    data["target_volatility"] = data.groupby("crypto_name")["current_volatility"].shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume", "marketCap",
    "current_volatility", "daily_return", "price_range_pct", "candle_body_pct",
    "volume_to_marketcap", "log_volume", "log_marketcap",
    "close_lag1", "close_pct_change_1d", "volume_lag1", "volume_pct_change_1d",
    "marketCap_lag1", "marketCap_pct_change_1d", "close_ma_3", "close_ma_7",
    "volatility_ma_3", "volatility_ma_7", "day_of_week", "month", "crypto_name"
]
TARGET_COLUMN = "target_volatility"
