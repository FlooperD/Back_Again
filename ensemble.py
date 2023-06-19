from binance.client import Client
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Set up the Binance API client
api_key = 'KEY'
api_secret = 'SECRET'
client = Client(api_key, api_secret)

# Retrieve historical price data
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
limit = 1000

klines = client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades_count', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# print(klines)
# print(df['close'].dtype)

# Perform feature engineering
window = 10
df['sma'] = SMAIndicator(close=df['close'], window=window).sma_indicator()

df['close'] = pd.to_numeric(df['close'])

# Calculate RSI
df['rsi'] = RSIIndicator(close=df['close']).rsi()

# Calculate MACD
df['macd'] = MACD(close=df['close']).macd()
df['macd_signal'] = MACD(close=df['close']).macd_signal()


# Split the data into training and testing sets
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Define the trading strategy
def strategy(data, i):
    if float(data['close'].iloc[i]) > float(data['sma'].iloc[i]) and float(data['rsi'].iloc[i]) < 70:
        return 'buy'
    elif float(data['close'].iloc[i]) < float(data['sma'].iloc[i]) and float(data['rsi'].iloc[i]) > 30:
        return 'sell'
    else:
        return 'hold'

# Define the backtesting function
def backtest(strategy, data):
    positions = []
    balances = [1.0]
    initial_balance = 1.0

    for i in range(len(data)):
        current_price = float(data['close'].iloc[i])
        action = strategy(data, i)

        if action == 'buy':
            if len(positions) == 0:
                positions.append((current_price, initial_balance))
                balances.append(initial_balance)
        elif action == 'sell':
            if len(positions) > 0:
                buy_price, buy_balance = positions.pop()
                buy_price = float(buy_price)  # Convert buy_price to float
                buy_balance = float(buy_balance)  # Convert buy_balance to float
                profit = (current_price - buy_price) / buy_price
                balances.append(buy_balance * (1 + profit))
            else:
                balances.append(initial_balance)
        else:
            balances.append(balances[-1])

    return balances

# Backtest the trading strategy
backtest_data = train_data.copy()
backtest_balances = backtest(strategy, backtest_data)
returns = (backtest_balances[-1] - backtest_balances[0]) / backtest_balances[0]
print(f"Backtesting returns: {returns}")

# Calculate additional performance metrics
returns = np.diff(backtest_balances) / backtest_balances[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns)
print(f"Sharpe ratio: {sharpe_ratio}")
cum_returns = np.cumprod(1 + returns) - 1
peak = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - peak) / (peak + 1e-9)
max_drawdown = np.min(drawdown)
print(f"Maximum drawdown: {max_drawdown}")

# Preprocess the data for machine learning
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(train_data[['sma', 'rsi', 'macd', 'macd_signal']])
y_train = train_data['close'].values
X_test = imputer.transform(test_data[['sma', 'rsi', 'macd', 'macd_signal']])

# Train the machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

for i in range(len(test_data)):
    current_price = float(test_data['close'].iloc[i])
    prediction = predictions[i]

    if prediction > current_price:
        # Buy logic
        print(f"Buy at {current_price}")
        # Execute buy order using Binance API
    elif prediction < current_price:
        # Sell logic
        print(f"Sell at {current_price}")
        # Execute sell order using Binance API
    else:
        # Hold logic
        print("Hold")
