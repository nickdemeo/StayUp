import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import datetime

# Load stock data
start = '2005-01-01'  # Ensure alignment with your extended date range
end = datetime.now().strftime('%Y-%m-%d')
df = yf.download("AAPL", start=start, end=end)  # Change "AAPL" if needed

# Keep only 'Close' price
df = df[['Close']]

# Split into training and testing sets (70% training, 30% testing)
train_size = int(len(df) * 0.70)
data_training = df.iloc[:train_size]
data_testing = df.iloc[train_size:]

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data sequences
x_train, y_train = [], []
seq_len = 100  # Sequence length

for i in range(seq_len, len(data_training_array)):
    x_train.append(data_training_array[i-seq_len: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(units=1))  # Output layer

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Save trained model
model.save('LSTM_model.keras', save_format='keras')

# Save the scaler for use in `main.py`
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
