# Crypto Volatility Prediction Project

## Objective
Predict **next-day crypto volatility** using current-day market information such as open, high, low, close, volume, market capitalization, rolling averages, and lag-based indicators.

## Project Structure

```text
crypto_volatility_project/
├── app.py
├── requirements.txt
├── data/
│   └── crypto_data.csv
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   └── FINAL_REPORT.md
├── models/
│   ├── volatility_model.joblib
│   └── metrics.json
├── notebooks/
│   └── eda.ipynb
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── predict.py
```

## Dataset
- Rows: 72,946
- Main columns: `open`, `high`, `low`, `close`, `volume`, `marketCap`, `crypto_name`, `date`

## Model
A `RandomForestRegressor` is used inside a preprocessing pipeline with:
- median imputation
- feature scaling for numeric features
- one-hot encoding for `crypto_name`

## Engineered Features
- current volatility
- daily return
- price range percentage
- candle body percentage
- volume-to-market-cap ratio
- log volume
- log market cap
- 1-day lag features
- 1-day percentage change features
- 3-day and 7-day moving averages
- day of week and month

## Evaluation
The project was trained using a time-based split.

- MAE: 0.034339
- RMSE: 0.154794
- R²: 0.055424
- Train rows: 55828
- Test rows: 13903
- Split date: 2021-07-30

## Setup

### 1. Create and activate virtual environment
On Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\activate
```

### 2. Install dependencies

pip install -r requirements.txt


### 3. Run Streamlit app

streamlit run app.py

