import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Data Collection
# Use your Binance API key to fetch historical price data for BTC pairs
from binance.client import Client

api_key = 'FEOKFzr1eTE5XkFM5q2Bdmu7k8KHbimFBnsbgYDZDY1W4qg2cpbZPgUYrKxitDin'
api_secret = 'L1yojgYtxG7HxlDiMI52KktMJ3XKgXvWq06WG1R8SJYSzMxcHRrHQLEIihVSd19P'
client = Client(api_key, api_secret)

# Specify the trading pair and time interval
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_30MINUTE

# Fetch historical price data
klines = client.futures_klines(symbol=symbol, interval=interval)

# Convert the data to a pandas DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
           'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
df = pd.DataFrame(klines, columns=columns)

# Extract the relevant columns
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set the timestamp as the DataFrame index
df.set_index('timestamp', inplace=True)

# Step 2: Data Preprocessing
# Clean and preprocess the data
# Split the data into training and testing sets

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# Split the data into train and test sets
train_size = int(len(normalized_data) * 0.8)
train_data = normalized_data[:train_size]
test_data = normalized_data[train_size:]

# Function to create sequences for the LSTM model
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Define the sequence length
sequence_length = 10

# Create sequences for training and testing
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Step 3: Model Architecture and Training
# Experiment with different model architectures and hyperparameters

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Step 4: Backtesting and Performance Metrics
# Evaluate the model on the testing data and calculate performance metrics

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Inverse transform the normalized predictions and actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate the mean absolute error (MAE)
mae = np.mean(np.abs(y_pred - y_test))
print(f'MAE: {mae}')

# Calculate the profit factor
gross_profit = np.sum(np.where(y_pred > 0, y_test - np.roll(y_test, 1), 0))
gross_loss = np.sum(np.where(y_pred < 0, np.roll(y_test, 1) - y_test, 0))
profit_factor = -gross_profit / gross_loss
print(f'Profit Factor: {profit_factor}')

# Calculate average risk per trade
average_risk_per_trade = np.mean(np.abs(y_test - np.roll(y_test, 1)))
print(f'Average Risk per Trade: {average_risk_per_trade}')

# Calculate average profit per trade
average_profit_per_trade = np.mean(np.where(y_pred > 0, y_test - np.roll(y_test, 1), 0))
print(f'Average Profit per Trade: {average_profit_per_trade}')

# Calculate winning percentage
winning_percentage = np.mean(np.where(y_pred > 0, 1, 0)) * 100
print(f'Winning Percentage: {winning_percentage}%')

# Plotting the results
train_size = len(train_data)
test_size = len(test_data)

# Create empty arrays for plotting
train_data_extended = np.concatenate([train_data, np.zeros((sequence_length, 1))])
test_data_extended = np.concatenate([train_data[-sequence_length:], test_data])

# Shift the predictions array to align with the data
y_pred_extended = np.concatenate([np.zeros((train_size, 1)), y_pred])

# Inverse transform the data and predictions
train_data_inverse = scaler.inverse_transform(train_data_extended)
test_data_inverse = scaler.inverse_transform(test_data_extended)
y_pred_inverse = scaler.inverse_transform(y_pred_extended)

# Create the x-axis for plotting
x_train = np.arange(train_size)
x_test = np.arange(train_size, train_size + test_size)

# Plot the training data
plt.plot(np.arange(train_size), train_data_inverse.flatten(), label='Training Data')

# Plot the testing data and predictions
plt.plot(np.arange(train_size, train_size + test_size), test_data_inverse.flatten()[:test_size], label='Testing Data')
plt.plot(np.arange(train_size, train_size + test_size), y_pred_inverse.flatten()[:test_size], label='Predictions')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BTC Price Prediction')
plt.legend()
plt.show()


client.close_connection()