import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model 
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import numpy as np
import pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


from tensorflow.keras.models import load_model


# Load model safely
try:
    model = load_model('LSTM_model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stops execution if model loading fails

# Page configuration
st.set_page_config(page_title='StayUp', layout='wide')
st.title('StayUp')
st.markdown("""
This interactive dashboard uses an LSTM model to predict stock prices based on Yahoo Finance data.
Select a stock ticker, determine a date range, and analyze the predicted price.
""")

# Sidebar Input
st.sidebar.header('User Input Features')
popular_stocks = ['TSLA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT', 'NVDA', 'META', 'NFLX', 'AMD', 'BA']
selected_stock = st.sidebar.selectbox("Choose Stock Ticker", popular_stocks)
start_date = st.sidebar.date_input("Start Date", datetime(2005, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Function to load stock data
@st.cache_data  # Caches data to avoid unnecessary re-fetching
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        st.error(f"No data found for {ticker}. Check ticker and date range.")
        return None  # Stops further execution if no data
    return data.drop(columns=['Adj Close'], errors='ignore')  # Drop 'Adj Close' if exists

# MinMaxScaler (defined globally for reuse)
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to preprocess and normalize data
def preprocess_data(data):
    return scaler.fit_transform(data[['Close']].values), scaler

# Optimized function to create time-series dataset
def create_dataset(dataset, time_step=100):
    return np.array([dataset[i-time_step:i, 0] for i in range(time_step, len(dataset))])

# Function to generate stock plot
def plot_stock_data(data, title="Historical Close Price", future_dates=None, predicted_prices=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.plot(data.index, data['Close'].rolling(50).mean(), label='50-Day MA', color='red')

    if future_dates is not None and predicted_prices is not None:
        ax.plot(future_dates, predicted_prices, label='Predicted Price', color='green', linestyle='--')

    ax.set(title=title, xlabel='Date', ylabel='Price (USD)')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend()
    return fig

# Load and display stock data
data = load_data(selected_stock, start_date, end_date)
if data is not None:
    st.write(f"Displaying data for: {selected_stock}")
    st.pyplot(plot_stock_data(data))

# Prediction Button
if st.button('Predict Future Prices'):  # Adding a button to trigger prediction
    scaled_data, scaler = preprocess_data(data)  # Preprocessing data for prediction
    x_test = create_dataset(scaled_data)  # Creating dataset for prediction
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)  # Reshaping data for LSTM model
    predictions = model.predict(x_test)  # Making predictions using the LSTM model
    predicted_prices = scaler.inverse_transform(predictions)  # Inverse transforming predicted prices

    # Generate future dates
    future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1, freq='B')[1:]

    # Plot predictions
    st.pyplot(plot_stock_data(data, title=f"Future Price Prediction for {selected_stock}", future_dates=future_dates, predicted_prices=predicted_prices))

# About Section
st.write("## About this Dashboard")
st.info("This dashboard analyzes historical stock data and predicts future trends using an LSTM model.")
