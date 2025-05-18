from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from alpha_vantage.timeseries import TimeSeries
import datetime

# --- Page Config ---
st.set_page_config(page_title="📈 LSTM Stock Predictor", layout="wide")

# --- Light/Dark Mode Toggle ---
theme = st.sidebar.radio("🌗 Choose Theme", ["Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
            body {background-color: #0e1117; color: #fafafa;}
            .stApp {background-color: #0e1117;}
            h1, h2, h3, h4, h5, h6 {color: #ffffff;}
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            body {background-color: #f8f9fa; color: #212529;}
            .stApp {background-color: #f8f9fa;}
        </style>
        """, unsafe_allow_html=True
    )

# --- Header ---
st.markdown("""
    <h1 style='text-align: center;'>📊 LSTM-Based Stock Price Predictor</h1>
    <p style='text-align: center; font-size: 18px;'>🚀 Predict future stock prices using Deep Learning (LSTM)!</p>
    <hr style='border: 1px solid #bbb;'>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Model Configuration")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TCS.BSE)", "AAPL")
api_key = st.sidebar.text_input("🔑 Enter Alpha Vantage API Key", type="password")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 7, 1)

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

        # --- Historical Data ---
        st.subheader(f"📈 Historical Stock Data for `{ticker}`")
        st.dataframe(data.tail(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        highest_price = float(data['Close'].max())
        lowest_price = float(data['Close'].min())
        start_price = float(data['Close'].iloc[0])
        end_price = float(data['Close'].iloc[-1])
        change_percent = ((end_price - start_price) / start_price) * 100

        col1.metric("Highest Price", f"₹{highest_price:.2f}")
        col2.metric("Lowest Price", f"₹{lowest_price:.2f}")
        col3.metric("Change Since Start", f"{change_percent:.2f}%")

        # --- Price Chart ---
        st.markdown("### 📉 Closing Price Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], data['Close'], label="Close Price", color="cyan" if theme == "Dark" else "blue")
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

        # --- LSTM Model ---
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # --- Training ---
        with st.spinner("🔁 Training LSTM model..."):
            history = model.fit(X, y, epochs=5, batch_size=64, verbose=0)

        # --- Model Summary ---
        with st.expander("🧠 View Model Summary"):
            summary_str = []
            model.summary(print_fn=lambda x: summary_str.append(x))
            st.text("\n".join(summary_str))

        # --- Loss Curve ---
        st.markdown("### 📉 Training Loss Curve")
        fig3, ax3 = plt.subplots()
        ax3.plot(history.history['loss'], color='red')
        ax3.set_title("Model Training Loss")
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("Loss")
        st.pyplot(fig3)

        # --- Prediction ---
        last_60 = scaled_data[-60:]
        future_preds = []

        for _ in range(days_to_predict):
            input_seq = last_60.reshape(1, 60, 1)
            pred = model.predict(input_seq, verbose=0)
            future_preds.append(pred[0][0])
            last_60 = np.append(last_60[1:], pred).reshape(-1, 1)

        predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # -- Model Accuracy using MAPE --
        mape_score = mean_absolute_percentage_error(y, model.predict(X))
        st.subheader("📏 Model Accuracy")
        st.info(f"Mean Absolute Percentage Error (MAPE): **{mape_score * 100:.2f}%**")

        st.markdown(f"### 🔮 Predicted Price for Next {days_to_predict} Day(s)")
        for i, price in enumerate(predicted_prices.flatten(), 1):
            st.success(f"📅 Day {i}: ₹{price:.2f}")

        

    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
else:
    st.warning("📌 Please enter both a stock symbol and a valid Alpha Vantage API key.")
