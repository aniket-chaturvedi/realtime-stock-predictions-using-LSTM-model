import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="LSTM Stock Predictor (CSV)", layout="wide")
st.title("LSTM-Based Stock Price Predictor (Upload CSV)")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with 'Date' and 'Close' columns", type=["csv"])

# Sidebar config
st.sidebar.header("Model Settings")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 7, 1)
epochs_to_train = st.sidebar.slider("Training Epochs", 20, 100, 30)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        df = df[['Date', 'Close']].dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        st.subheader("Uploaded Data")
        st.dataframe(df.tail(), use_container_width=True)

        # Visualization
        st.markdown("### Close Price Over Time")
        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df['Close'], color='blue')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        st.pyplot(plt.gcf())

        # Preprocessing
        close_data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        X, y = [], []
        for i in range(60, len(scaled_data) - days_to_predict):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i+days_to_predict-1])

        X, y = np.array(X), np.array(y)

        X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
        y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

        # Model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs_to_train, batch_size=64, verbose=0)

        # Evaluate
        y_pred = model.predict(X_test)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_unscaled = scaler.inverse_transform(y_pred)
        mape = mean_absolute_percentage_error(y_test_unscaled, y_pred_unscaled)

        st.success(f"Model Test MAPE: {mape * 100:.2f}%")

        # Future prediction
        st.markdown(f"### Predicted Price for Next {days_to_predict} Day(s)")
        last_60 = scaled_data[-60:]
        future_preds = []

        for _ in range(days_to_predict):
            input_seq = last_60.reshape(1, 60, 1)
            pred = model.predict(input_seq, verbose=0)
            future_preds.append(pred[0][0])
            last_60 = np.concatenate((last_60[1:], pred.reshape(1, 1)), axis=0)

        predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        for i, price in enumerate(predicted_prices.flatten(), 1):
            st.info(f"Day {i}: Rs. {price:.2f}")
