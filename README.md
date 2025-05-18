# 📈 LSTM Stock Price Prediction Web App

A real-time stock price prediction web application built using **LSTM (Long Short-Term Memory)** neural networks and **Streamlit**. This project predicts future stock prices based on historical data and evaluates model performance using **K-Fold Cross Validation**.

## 🔧 Features

- 🧠 Deep Learning-based prediction using LSTM
- 🔁 K-Fold Cross Validation for robust model evaluation
- 📊 Interactive charts to visualize:
  - Closing Price Trends
  - Training Loss per Epoch
  - Mean Absolute Percentage Error (MAPE)
- 📅 Adjustable forecasting range (1 to 7 days)
- 🛠️ Customizable training parameters (epochs, folds)
- 🔐 Secure API key input for fetching real-time stock data via Alpha Vantage

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/Model:** Keras (TensorFlow), LSTM
- **Data Fetching:** Alpha Vantage API
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib
- **Model Evaluation:** Scikit-learn (`mean_absolute_percentage_error`, `KFold`)

---

## 📂 Folder Structure
- lstm.py # Main Streamlit app
├── README.md # Project README
├── requirements.txt # Python dependencies

