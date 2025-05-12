This is my training model that works
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Generate some sample data
data = np.sin(np.linspace(0, 100, 1000))  
df = pd.DataFrame(data, columns=["Value"])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df["Scaled_Value"] = scaler.fit_transform(df[["Value"]])

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(df["Scaled_Value"].values, sequence_length)

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X, y, epochs=10, batch_size=1, verbose=1)

# Save the model
model.save("LSTM_model.h5")

# Verify model creation
if os.path.exists("LSTM_model.h5"):
    print("✅ Model successfully created and saved as 'LSTM_model.h5'")
else:
    print("❌ Model not saved correctly!")


