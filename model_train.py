# This file can be used to train the ML models

import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Define constants
DATA_DIR = 'stock_data'  # Directory containing CSV files
OUTPUT_DIR = 'model_results'  # Directory to save results
PREDICTION_MONTHS = 3  # Number of months to predict
TRAIN_TEST_SPLIT = 0.8  # Proportion of data for training
RANDOM_SEED = 42  # For reproducibility

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_stock_data(symbol=None):
    """
    Load stock data from CSV files
    
    Parameters:
    -----------
    symbol: str or None
        If provided, loads data only for that symbol.
        If None, loads all available CSV files in the data directory.
    
    Returns:
    --------
    dict: Dictionary where keys are symbols and values are pandas DataFrames
    """
    stock_data = {}
    
    if symbol:
        pattern = f"{DATA_DIR}/{symbol}_monthly_adjusted_*.csv"
    else:
        pattern = f"{DATA_DIR}/*_monthly_adjusted_*.csv"
    
    for file_path in glob.glob(pattern):
        # Extract symbol from the file name
        file_name = os.path.basename(file_path)
        current_symbol = file_name.split('_')[0]
        
        # Load CSV
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Handle multiple files for the same symbol
        if current_symbol in stock_data:
            # Use the most recent file (assuming timestamp is in the filename)
            prev_file_name = os.path.basename(stock_data[current_symbol]['file_path'])
            if file_name > prev_file_name:
                stock_data[current_symbol] = {'data': df, 'file_path': file_path}
        else:
            stock_data[current_symbol] = {'data': df, 'file_path': file_path}
    
    return {k: v['data'] for k, v in stock_data.items()}

def create_features(df):
    """
    Create features for ML models
    
    Parameters:
    -----------
    df: pandas DataFrame
        Stock data with 'adj_close' column
    
    Returns:
    --------
    pandas DataFrame: Data with additional features
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Basic price features
    data['return'] = data['adj_close'].pct_change()
    data['return_lag1'] = data['return'].shift(1)
    data['return_lag2'] = data['return'].shift(2)
    data['return_lag3'] = data['return'].shift(3)
    
    # Moving averages
    data['ma3'] = data['adj_close'].rolling(window=3).mean()
    data['ma6'] = data['adj_close'].rolling(window=6).mean()
    data['ma12'] = data['adj_close'].rolling(window=12).mean()
    
    # Price differences from moving averages
    data['ma3_diff'] = (data['adj_close'] - data['ma3']) / data['adj_close']
    data['ma6_diff'] = (data['adj_close'] - data['ma6']) / data['adj_close']
    data['ma12_diff'] = (data['adj_close'] - data['ma12']) / data['adj_close']
    
    # Volatility (standard deviation over rolling window)
    data['volatility_3'] = data['return'].rolling(window=3).std()
    data['volatility_6'] = data['return'].rolling(window=6).std()
    data['volatility_12'] = data['return'].rolling(window=12).std()
    
    # Momentum indicators
    data['momentum_3'] = data['adj_close'] / data['adj_close'].shift(3) - 1
    data['momentum_6'] = data['adj_close'] / data['adj_close'].shift(6) - 1
    data['momentum_12'] = data['adj_close'] / data['adj_close'].shift(12) - 1
    
    # Month of the year (to capture seasonality)
    data['month'] = data.index.month
    
    # Year (to capture long-term trends)
    data['year'] = data.index.year
    
    # One-hot encode month
    month_dummies = pd.get_dummies(data['month'], prefix='month', drop_first=True)
    data = pd.concat([data, month_dummies], axis=1)
    
    # Create target variables for different prediction horizons
    for i in range(1, PREDICTION_MONTHS + 1):
        data[f'target_{i}m'] = data['adj_close'].shift(-i)
        data[f'target_return_{i}m'] = data[f'target_{i}m'] / data['adj_close'] - 1
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

def prepare_data_for_xgboost(data, target_column, test_size=0.2):
    """
    Prepare data for XGBoost model
    """
    # Define features to use
    features = [
        'return', 'return_lag1', 'return_lag2', 'return_lag3',
        'ma3_diff', 'ma6_diff', 'ma12_diff',
        'volatility_3', 'volatility_6', 'volatility_12',
        'momentum_3', 'momentum_6', 'momentum_12'
    ]
    
    # Add month dummies
    month_cols = [col for col in data.columns if col.startswith('month_')]
    features.extend(month_cols)
    
    # Split data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Create feature matrices and target vectors
    X_train = train_data[features]
    y_train = train_data[target_column]
    X_test = test_data[features]
    y_test = test_data[target_column]
    
    return X_train, X_test, y_train, y_test, features

def build_and_train_xgboost(X_train, y_train, X_test, y_test):
    """
    Build and train XGBoost model
    """
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 500,
        'random_state': RANDOM_SEED
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"XGBoost Train RMSE: {train_rmse:.4f}")
    print(f"XGBoost Test RMSE: {test_rmse:.4f}")
    print(f"XGBoost Test MAE: {test_mae:.4f}")
    print(f"XGBoost Test R²: {test_r2:.4f}")
    
    # Return model and predictions
    return model, test_preds, (train_rmse, test_rmse, test_mae, test_r2)

def prepare_data_for_lstm(data, target_column, test_size=0.2, seq_length=12):
    """
    Prepare data for LSTM model
    """
    # Select features for LSTM
    features = [
        'adj_close', 'return', 'ma3_diff', 'ma6_diff', 'ma12_diff',
        'volatility_3', 'momentum_3', 'momentum_6', 'momentum_12'
    ]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - PREDICTION_MONTHS + 1):
        X.append(scaled_data[i:i+seq_length])
        y_idx = i + seq_length + (int(target_column.split('_')[1][0]) - 1)
        if target_column.startswith('target_return'):
            # For return targets
            y.append(data[target_column].iloc[i+seq_length])
        else:
            # For price targets
            y.append(data[target_column].iloc[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, features

def build_and_train_lstm(X_train, y_train, X_test, y_test):
    """
    Build and train LSTM model
    """
    # Set up model architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"LSTM Train RMSE: {train_rmse:.4f}")
    print(f"LSTM Test RMSE: {test_rmse:.4f}")
    print(f"LSTM Test MAE: {test_mae:.4f}")
    print(f"LSTM Test R²: {test_r2:.4f}")
    
    # Return model, predictions, and metrics
    return model, test_preds, (train_rmse, test_rmse, test_mae, test_r2), history

def prepare_data_for_prophet(data, target_column):
    """
    Prepare data for Prophet model
    """
    # For Prophet, we need 'ds' (dates) and 'y' (target)
    prophet_data = pd.DataFrame({
        'ds': data.index,
        'y': data['adj_close'],
        'volume': data['volume'] if 'volume' in data.columns else np.zeros(len(data)),
        'return': data['return'],
        'volatility': data['volatility_6']
    })
    
    # Split chronologically
    split_idx = int(len(prophet_data) * TRAIN_TEST_SPLIT)
    train_data = prophet_data.iloc[:split_idx]
    test_data = prophet_data.iloc[split_idx:]
    
    return train_data, test_data

def build_and_train_prophet(train_data, test_data):
    """
    Build and train a Prophet model with custom components
    """
    # Initialize model with custom seasonality parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add monthly seasonality (important for stocks)
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    # Add additional regressor features
    model.add_regressor('volume', standardize=True)
    model.add_regressor('return', standardize=True)
    model.add_regressor('volatility', standardize=True)
    
    # Fit model
    print("Training Prophet model...")
    model.fit(train_data)
    
    # Create future dataframe for testing period
    future = test_data.copy()
    
    # Predict
    forecast = model.predict(future)
    
    # Evaluate
    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Prophet Test RMSE: {rmse:.4f}")
    print(f"Prophet Test MAE: {mae:.4f}")
    print(f"Prophet Test R²: {r2:.4f}")
    
    # Make future predictions
    future_periods = model.make_future_dataframe(periods=PREDICTION_MONTHS, freq='M')
    
    # Add regressor values for future periods (use averages from training data)
    future_periods['volume'] = train_data['volume'].mean()
    future_periods['return'] = train_data['return'].mean()
    future_periods['volatility'] = train_data['volatility'].mean()
    
    # Predict future
    future_forecast = model.predict(future_periods)
    
    return model, forecast, future_forecast, (rmse, mae, r2)

def predict_future_xgboost(model, features, last_data_point, months_ahead=3):
    """
    Recursively predict future values with XGBoost
    """
    predictions = []
    current_data = last_data_point.copy()
    
    for i in range(months_ahead):
        # Predict next month
        next_pred = model.predict(current_data[features].values.reshape(1, -1))[0]
        predictions.append(next_pred)
        
        # Update features for next prediction (this is simplified; in reality
        # you'd need to update all features based on the new prediction)
        # For this example, we're just updating the last few features
        if 'return' in features:
            current_data['return'] = next_pred
        
        if 'return_lag1' in features:
            current_data['return_lag1'] = current_data['return']
        
        if 'return_lag2' in features:
            current_data['return_lag2'] = current_data['return_lag1']
    
    return predictions

def plot_results(symbol, data, test_data, xgb_preds, lstm_preds, prophet_forecast, future_forecast, target_column):
    """
    Plot the results from all models
    """
    plt.figure(figsize=(14, 7))
    
    # Plot actual data
    plt.plot(data.index, data['adj_close'], label='Historical Data', color='black')
    
    # Plot test predictions for each model
    if xgb_preds is not None:
        # Assuming test_data has the same index as the test_data used for XGBoost
        plt.plot(test_data.index[-len(xgb_preds):], xgb_preds, 
                label='XGBoost Predictions', linestyle='--', color='blue')
    
    if lstm_preds is not None:
        # For LSTM, we need to map predictions back to dates
        plt.plot(test_data.index[-len(lstm_preds):], lstm_preds, 
                label='LSTM Predictions', linestyle='--', color='green')
    
    if prophet_forecast is not None:
        # For Prophet, we use the forecast dataframe
        plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], 
                label='Prophet Predictions', linestyle='--', color='red')
        
        # Add prediction intervals
        plt.fill_between(prophet_forecast['ds'], 
                        prophet_forecast['yhat_lower'], 
                        prophet_forecast['yhat_upper'], 
                        color='red', alpha=0.1)
    
    # Plot future predictions if available
    if future_forecast is not None:
        # Get only the future dates (those beyond the historical data)
        future_only = future_forecast[future_forecast['ds'] > data.index[-1]]
        
        plt.plot(future_only['ds'], future_only['yhat'], 
                label='Prophet Future Forecast', linestyle=':', color='purple')
        
        # Add prediction intervals for future
        plt.fill_between(future_only['ds'], 
                        future_only['yhat_lower'], 
                        future_only['yhat_upper'], 
                        color='purple', alpha=0.1)
    
    # Add labels and title
    plt.title(f'{symbol} Stock Price Prediction ({PREDICTION_MONTHS} Months Ahead)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = os.path.join(OUTPUT_DIR, f"{symbol}_prediction_comparison.png")
    plt.savefig(filename)
    plt.close()
    
    return filename

def evaluate_models(symbol, data, xgb_metrics, lstm_metrics, prophet_metrics):
    """
    Create a comparison of model metrics
    """
    models = ['XGBoost', 'LSTM', 'Prophet']
    metrics_df = pd.DataFrame({
        'Model': models,
        'RMSE': [xgb_metrics[1], lstm_metrics[1], prophet_metrics[0]],
        'MAE': [xgb_metrics[2], lstm_metrics[2], prophet_metrics[1]],
        'R²': [xgb_metrics[3], lstm_metrics[3], prophet_metrics[2]]
    })
    
    # Sort by RMSE (lower is better)
    metrics_df = metrics_df.sort_values('RMSE')
    
    # Save to CSV
    filename = os.path.join(OUTPUT_DIR, f"{symbol}_model_comparison.csv")
    metrics_df.to_csv(filename, index=False)
    
    # Create a bar chart for RMSE comparison
    plt.figure(figsize=(10, 6))
    
    plt.bar(models, metrics_df['RMSE'], color=['blue', 'green', 'red'])
    
    plt.title(f'Model RMSE Comparison for {symbol}')
    plt.xlabel('Model')
    plt.ylabel('RMSE (Lower is Better)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    chart_filename = os.path.join(OUTPUT_DIR, f"{symbol}_model_comparison.png")
    plt.savefig(chart_filename)
    plt.close()
    
    return metrics_df, chart_filename

def run_all_models(symbol, data):
    """
    Run all models for a given stock symbol
    """
    print(f"\n{'='*50}")
    print(f"Processing {symbol} with {len(data)} data points")
    print(f"{'='*50}")
    
    # Add features
    processed_data = create_features(data)
    print(f"Created features. Shape: {processed_data.shape}")
    
    # Define target column (3-month ahead price)
    target_column = f'target_{PREDICTION_MONTHS}m'
    
    # MODEL 1: XGBOOST
    print("\n1. Training XGBoost model...")
    X_train, X_test, y_train, y_test, features = prepare_data_for_xgboost(
        processed_data, target_column)
    
    xgb_model, xgb_preds, xgb_metrics = build_and_train_xgboost(
        X_train, y_train, X_test, y_test)
    
    # MODEL 2: LSTM
    print("\n2. Training LSTM model...")
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm, lstm_features = prepare_data_for_lstm(
        processed_data, target_column)
    
    lstm_model, lstm_preds, lstm_metrics, lstm_history = build_and_train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    
    # MODEL 3: PROPHET
    print("\n3. Training Prophet model...")
    train_prophet, test_prophet = prepare_data_for_prophet(processed_data, target_column)
    
    prophet_model, prophet_forecast, future_forecast, prophet_metrics = build_and_train_prophet(
        train_prophet, test_prophet)
    
    # Compare and plot results
    print("\nPlotting results...")
    plot_file = plot_results(
        symbol, data, processed_data, 
        xgb_preds, lstm_preds.flatten(), 
        prophet_forecast, future_forecast, 
        target_column
    )
    
    # Compare metrics
    metrics_df, metrics_chart = evaluate_models(
        symbol, data, xgb_metrics, lstm_metrics, prophet_metrics)
    
    print("\nResults saved to:")
    print(f"- Plot: {plot_file}")
    print(f"- Metrics: {metrics_chart}")
    
    return {
        'xgboost': (xgb_model, xgb_metrics),
        'lstm': (lstm_model, lstm_metrics),
        'prophet': (prophet_model, prophet_metrics),
        'metrics': metrics_df
    }

def main():
    """
    Main function to run all models on all available stock data
    """
    print("Stock Price Prediction with Multiple ML Models")
    print("=============================================")
    
    # Load all available stock data
    all_stock_data = load_stock_data()
    
    if not all_stock_data:
        print("No stock data found. Please make sure CSV files exist in the data directory.")
        return
    
    print(f"Found data for {len(all_stock_data)} stocks: {', '.join(all_stock_data.keys())}")
    
    # Run models for each stock
    results = {}
    
    for symbol, data in all_stock_data.items():
        results[symbol] = run_all_models(symbol, data)
    
    print("\nAll models completed!")

if __name__ == "__main__":
    main()