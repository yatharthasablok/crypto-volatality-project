# High-Level Design (HLD)

## 1. Problem Statement
The goal is to predict next-day cryptocurrency volatility using historical market data.

## 2. Input Data
Historical crypto records containing:
- open
- high
- low
- close
- volume
- marketCap
- crypto_name
- date

## 3. Solution Components
1. **Data ingestion**
   - Read the CSV dataset.
   - Parse dates and sort by crypto and date.

2. **Preprocessing and feature engineering**
   - Create volatility, returns, lag features, moving averages, and seasonal features.

3. **Model training**
   - Train a regression model to predict next-day volatility.

4. **Evaluation**
   - Compute MAE, RMSE, and R².

5. **Deployment**
   - Provide a Streamlit web app to perform predictions.

## 4. High-Level Architecture
```text
CSV Dataset
   ↓
Preprocessing + Feature Engineering
   ↓
Train/Test Split (Time-Based)
   ↓
Random Forest Regressor
   ↓
Saved Model (.joblib)
   ↓
Streamlit App
```

## 5. Expected Output
A predicted next-day volatility score for a selected cryptocurrency.
