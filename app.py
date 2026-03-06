from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURE_COLUMNS, build_features, load_data

DATA_PATH = "data/crypto_data.csv"
MODEL_PATH = "models/volatility_model.joblib"
METRICS_PATH = "models/metrics.json"

st.set_page_config(page_title="Crypto Volatility Prediction", layout="wide")
st.title("📈 Crypto Next-Day Volatility Prediction")
st.write("Use current-day market information to estimate the next day's volatility for a selected cryptocurrency.")

@st.cache_data
def get_processed_data() -> pd.DataFrame:
    raw = load_data(DATA_PATH)
    return build_features(raw)

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

df = get_processed_data()
model = get_model()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

st.sidebar.header("Model Summary")
st.sidebar.write(f"**MAE:** {metrics['mae']:.4f}")
st.sidebar.write(f"**RMSE:** {metrics['rmse']:.4f}")
st.sidebar.write(f"**R²:** {metrics['r2']:.4f}")
st.sidebar.write(f"Train rows: {metrics['train_rows']:,}")
st.sidebar.write(f"Test rows: {metrics['test_rows']:,}")
st.sidebar.write(f"Time split date: {metrics['split_date']}")

crypto_options = sorted(df["crypto_name"].unique().tolist())
selected_crypto = st.selectbox("Crypto name", crypto_options)

latest_row = df[df["crypto_name"] == selected_crypto].sort_values("date").iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    open_price = st.number_input("Open", value=float(latest_row["open"]))
    high_price = st.number_input("High", value=float(latest_row["high"]))
    low_price = st.number_input("Low", value=float(latest_row["low"]))
    close_price = st.number_input("Close", value=float(latest_row["close"]))

with col2:
    volume = st.number_input("Volume", value=float(latest_row["volume"]), format="%.6f")
    market_cap = st.number_input("Market Cap", value=float(latest_row["marketCap"]), format="%.6f")
    close_lag1 = st.number_input("Previous Close", value=float(latest_row["close_lag1"]))
    volume_lag1 = st.number_input("Previous Volume", value=float(latest_row["volume_lag1"]), format="%.6f")

with col3:
    marketcap_lag1 = st.number_input("Previous Market Cap", value=float(latest_row["marketCap_lag1"]), format="%.6f")
    day_of_week = st.selectbox("Day of Week", list(range(7)), index=int(latest_row["day_of_week"]))
    month = st.selectbox("Month", list(range(1, 13)), index=int(latest_row["month"]) - 1)

# derived features
eps = 1e-9
current_volatility = (high_price - low_price) / (abs(close_price) + eps)
daily_return = (close_price - open_price) / (abs(open_price) + eps)
price_range_pct = (high_price - low_price) / (abs(open_price) + eps)
candle_body_pct = abs(close_price - open_price) / (abs(open_price) + eps)
volume_to_marketcap = volume / (abs(market_cap) + eps)
log_volume = float(pd.Series([max(volume, 0)]).map(lambda x: __import__('numpy').log1p(x)).iloc[0])
log_marketcap = float(pd.Series([max(market_cap, 0)]).map(lambda x: __import__('numpy').log1p(x)).iloc[0])

# use recent row for rolling / pct-change based fields; update what we can from current inputs
close_pct_change_1d = (close_price - close_lag1) / (abs(close_lag1) + eps)
volume_pct_change_1d = (volume - volume_lag1) / (abs(volume_lag1) + eps)
marketcap_pct_change_1d = (market_cap - marketcap_lag1) / (abs(marketcap_lag1) + eps)

close_ma_3 = float(latest_row["close_ma_3"])
close_ma_7 = float(latest_row["close_ma_7"])
volatility_ma_3 = float(latest_row["volatility_ma_3"])
volatility_ma_7 = float(latest_row["volatility_ma_7"])

sample = pd.DataFrame([{
    "open": open_price,
    "high": high_price,
    "low": low_price,
    "close": close_price,
    "volume": volume,
    "marketCap": market_cap,
    "current_volatility": current_volatility,
    "daily_return": daily_return,
    "price_range_pct": price_range_pct,
    "candle_body_pct": candle_body_pct,
    "volume_to_marketcap": volume_to_marketcap,
    "log_volume": log_volume,
    "log_marketcap": log_marketcap,
    "close_lag1": close_lag1,
    "close_pct_change_1d": close_pct_change_1d,
    "volume_lag1": volume_lag1,
    "volume_pct_change_1d": volume_pct_change_1d,
    "marketCap_lag1": marketcap_lag1,
    "marketCap_pct_change_1d": marketcap_pct_change_1d,
    "close_ma_3": close_ma_3,
    "close_ma_7": close_ma_7,
    "volatility_ma_3": volatility_ma_3,
    "volatility_ma_7": volatility_ma_7,
    "day_of_week": day_of_week,
    "month": month,
    "crypto_name": selected_crypto,
}])[FEATURE_COLUMNS]

if st.button("Predict next-day volatility", type="primary"):
    prediction = float(model.predict(sample)[0])
    st.success(f"Predicted next-day volatility: {prediction:.6f}")
