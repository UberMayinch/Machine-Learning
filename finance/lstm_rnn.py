import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import Adam
import os
import yfinance as yf

# Download the stock data if not already present
if not os.path.exists('MSFT.csv'):
    msft = yf.Ticker('MSFT')
    msft_hist = msft.history(period='max')
    msft_hist.to_csv('MSFT.csv')
import matplotlib.pyplot as plt

# Load the stock data
data = pd.read_csv('MSFT.csv')
data = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(5, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Build the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)))
rnn_model.add(SimpleRNN(50, return_sequences=False))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the models
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions
lstm_predictions = lstm_model.predict(X_test)
rnn_predictions = rnn_model.predict(X_test)

# Inverse transform the predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions)
rnn_predictions = scaler.inverse_transform(rnn_predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(lstm_predictions, color='red', label='LSTM Predicted Stock Price')
plt.plot(rnn_predictions, color='green', label='RNN Predicted Stock Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()