from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries

# --- Page Config ---
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# --- Header ---
st.markdown("""
    <h1 style='text-align: center;'>LSTM-Based Stock Price Predictor</h1>
    <p style='text-align: center; font-size: 18px;'>Predict future stock prices using Deep Learning (LSTM)!</p>
    <hr style='border: 1px solid #bbb;'>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Model Configuration")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TCS.BSE)", "AAPL")
api_key = st.sidebar.text_input("Enter Alpha Vantage API Key", type="password")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 7, 1)
k_folds = st.sidebar.slider("K-Folds for Cross-Validation", 2, 5, 3)
epochs_to_train = st.sidebar.slider("Training Epochs", 20, 100, 20)

# --- Load Data ---
@st.cache_data
def load_data_alpha(ticker, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.sort_index()
    data.reset_index(inplace=True)
    data.rename(columns={
        'date': 'Date',
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)
    return data

if ticker and api_key:
    try:
        data = load_data_alpha(ticker, api_key)

        st.subheader(f"Historical Stock Data for {ticker}")
        st.dataframe(data.tail(), use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        highest_price = float(data['Close'].max())
        lowest_price = float(data['Close'].min())

        data_sorted_by_date = data.sort_values(by="Date")
        start_price = float(data_sorted_by_date['Close'].iloc[0])
        end_price = float(data_sorted_by_date['Close'].iloc[-1])
        change_percent = ((end_price - start_price) / start_price) * 100

        col1.metric("Highest Price", f"Rs. {highest_price:.2f}")
        col2.metric("Lowest Price", f"Rs. {lowest_price:.2f}")
        col3.metric("Change Since Start", f"{change_percent:.2f}%")

        # Chart
        st.markdown("### Closing Price Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], data['Close'], label="Close Price", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

        # --- Preprocessing ---
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(60, len(scaled_data) - days_to_predict):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i+days_to_predict-1])

        X, y = np.array(X), np.array(y)

        # Train/Test Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- Model Training with K-Fold ---
        st.markdown("### Training Model with K-Fold Cross Validation")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []
        history_per_fold = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            X_train_fold = X_train[train_idx].reshape(-1, 60, 1)
            X_val_fold = X_train[val_idx].reshape(-1, 60, 1)
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            history = model.fit(X_train_fold, y_train_fold, epochs=epochs_to_train, batch_size=64, verbose=1)
            history_per_fold.append(history)

            val_pred = model.predict(X_val_fold, verbose=0)
            val_true = scaler.inverse_transform(y_val_fold.reshape(-1, 1))
            val_pred_unscaled = scaler.inverse_transform(val_pred)
            val_mape = mean_absolute_percentage_error(val_true, val_pred_unscaled)
            fold_scores.append(val_mape)

        st.info(f"Average MAPE across {k_folds} folds: {np.mean(fold_scores) * 100:.2f}%")

        # Plot Epoch Loss Chart
        st.markdown("### Training Loss per Epoch")
        plt.figure(figsize=(10, 4))
        for i, hist in enumerate(history_per_fold):
            plt.plot(hist.history['loss'], label=f"Fold {i+1}")
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        # Final Training on Full Train Set
        model.fit(X_train, y_train, epochs=epochs_to_train, batch_size=64, verbose=0)

        # --- Prediction ---
        last_60 = scaled_data[-60:]
        future_preds = []

        for _ in range(days_to_predict):
            input_seq = last_60.reshape(1, 60, 1)
            pred = model.predict(input_seq, verbose=0)
            future_preds.append(pred[0][0])
            last_60 = np.concatenate((last_60[1:], pred.reshape(1, 1)), axis=0)

        predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        test_pred = model.predict(X_test)
        test_true = scaler.inverse_transform(y_test.reshape(-1, 1))
        test_pred_unscaled = scaler.inverse_transform(test_pred)
        test_mape = mean_absolute_percentage_error(test_true, test_pred_unscaled)

        st.subheader("Model Accuracy")
        st.info(f"Test MAPE: {test_mape * 100:.2f}%")

        st.markdown(f"### Predicted Price for Next {days_to_predict} Day(s)")
        for i, price in enumerate(predicted_prices.flatten(), 1):
            st.success(f"Day {i}: Rs. {price:.2f}")

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.warning("Please enter both a stock symbol and a valid Alpha Vantage API key.")
