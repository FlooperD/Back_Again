import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Step 1: Data Collection
# Use your Binance API key to fetch historical price data for BTC pairs
from binance.client import Client
import pandas as pd

api_key = 'KEY'
api_secret = 'SECRET'
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
df.to_csv("verify.csv")

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

# Step 3: Feature Engineering
# Create relevant features based on the data, such as technical indicators

# No feature engineering example provided. You can add your own code here to create additional features.

# Step 4: Model Training
# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 5: Backtesting
# Use the trained model to make predictions on the testing data

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Inverse transform the normalized predictions and actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate the mean absolute error (MAE) as a performance metric
mae = np.mean(np.abs(y_pred - y_test))
print(f'MAE: {mae}')

# Calculate additional performance metrics
returns = y_test[1:] - y_test[:-1]
gross_profit = np.sum(np.where(returns > 0, returns, 0))
gross_loss = np.sum(np.where(returns < 0, returns, 0))
profit_factor = -gross_profit / (gross_loss + 1e-8)  # Add a small value to avoid division by zero
average_risk_per_trade = np.abs(returns).mean()
average_profit_per_trade = np.mean(np.where(returns > 0, returns, 0))

# Calculate winning percentage
num_winning_trades = np.sum(np.where(returns > 0, 1, 0))
num_losing_trades = np.sum(np.where(returns < 0, 1, 0))
total_trades = num_winning_trades + num_losing_trades
winning_percentage = num_winning_trades / total_trades * 100

# Print the additional performance metrics
print(f'Profit Factor: {profit_factor}')
print(f'Average Risk per Trade: {average_risk_per_trade}')
print(f'Average Profit per Trade: {average_profit_per_trade}')
print(f'Winning Percentage: {winning_percentage}%')

client.close_connection()
