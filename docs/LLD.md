# Low-Level Design (LLD)

## Module 1: `src/data_preprocessing.py`
### Responsibilities
- load CSV data
- convert date column to datetime
- generate engineered features
- create target variable (`target_volatility`)

### Important Methods
- `load_data(path)`
- `build_features(df)`

## Module 2: `src/model_training.py`
### Responsibilities
- build train/test split using time
- create preprocessing pipeline
- train RandomForestRegressor
- calculate metrics
- save model and metrics

### Important Methods
- `train_model(data_path, model_output, metrics_output)`

## Module 3: `src/predict.py`
### Responsibilities
- load saved model
- accept one record of feature inputs
- return predicted next-day volatility

## Module 4: `app.py`
### Responsibilities
- load processed data and trained model
- take user input
- derive required features from UI values
- show prediction result in Streamlit

## Data Flow
1. Raw CSV is loaded.
2. Features are engineered.
3. Data is split chronologically.
4. Model is trained and saved.
5. Streamlit app loads the model and performs inference.

## Why time-based split?
Random shuffling can leak future information in financial time series. Time-based splitting is safer for evaluation.
