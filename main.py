import streamlit as st  # Importing Streamlit library for creating web apps
import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
from keras.models import load_model  # type: ignore # Importing Keras for loading the LSTM model
import yfinance as yf  # Importing yfinance for fetching stock data
from datetime import datetime  # Importing datetime for handling dates
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for data normalization
import matplotlib.dates as mdates  # Importing matplotlib.dates for date formatting

# Load model
model = load_model('LSTM_model.h5')  # Loading the pre-trained LSTM model

# Page configuration
st.set_page_config(page_title='InvestIQ', layout='wide')  # Configuring the Streamlit page layout and title
# Title and introduction
st.title('InvestIQ')  # Adding a title to the Streamlit app
st.markdown("""
This interactive dashboard uses a Long Short-Term Memory (LSTM) network to predict stock prices based on historical data from Yahoo Finance.
Select a stock ticker, define the date range, and click the predict button to see future price projections.
""")  # Adding a markdown text for introduction

# Sidebar - User input features
st.sidebar.header('User Input Features')  # Adding a header to the sidebar
selected_stock = st.sidebar.text_input("Enter Stock Ticker", 'TSLA')  # Adding a text input for entering stock ticker in the sidebar
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))  # Adding a date input for selecting start date in the sidebar
end_date = st.sidebar.date_input("End Date", datetime.now())  # Adding a date input for selecting end date in the sidebar
