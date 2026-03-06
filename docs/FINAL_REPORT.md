# Final Report: Crypto Volatility Prediction

## 1. Introduction
Cryptocurrency markets are highly volatile. The objective of this project is to predict the next day's volatility using historical trading and market capitalization data.


## 2. Dataset Description
The dataset contains 72,946 rows across multiple cryptocurrencies. Key variables include:
- open
- high
- low
- close
- volume
- marketCap
- crypto_name
- date

## 3. Data Preprocessing
The following preprocessing steps were applied:
- date parsing
- sorting by `crypto_name` and `date`
- removal of missing and infinite values
- generation of lag and rolling-window features

## 4. Feature Engineering
Created features:
- current volatility = `(high - low) / close`
- daily return = `(close - open) / open`
- price range percentage
- candle body percentage
- volume to market cap ratio
- log(volume), log(marketCap)
- lag features for close, volume, marketCap
- 3-day and 7-day moving averages

## 5. Target Variable
The target is **next-day volatility**, computed by shifting current volatility by one step within each cryptocurrency.

## 6. Model Used
A Random Forest Regressor was chosen because it is:
- easy to train
- robust to non-linear relationships
- suitable for mixed-scale tabular data

## 7. Evaluation Metrics
Using a time-based split:
- MAE: 0.034339
- RMSE: 0.154794
- R²: 0.055424

## 8. Deployment
A Streamlit app was developed to:
- choose a cryptocurrency
- input market values
- predict next-day volatility

## 9. Conclusion
The project provides a full machine learning pipeline from raw data to deployment. It can be extended with stronger time-series models, hyperparameter tuning, and more advanced financial indicators.
