import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler

# 📌 Load Trained Model
model = joblib.load("stock_price_predictor.pkl")

# 📌 Stock Data Fetch Karna
ticker = "AAPL"
df = yf.download(ticker, start="2025-01-02", end="2025-03-28")  # Latest data le rahe hain

# 📌 Feature Engineering (Same as Training)
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['Volatility'] = df['High'] - df['Low']
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# 📌 Select Features
X_new = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'EMA_10', 'Volatility', 'RSI']]

# 📌 Feature Scaling
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# 📌 Predict Future Prices
predicted_prices = model.predict(X_new)

# 📌 Show Predictions
df['Predicted_Close'] = predicted_prices
print(df[['Close', 'Predicted_Close']].tail(10))  # Last 10 predictions print karega

# 📌 Graph Plot Karna
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label="Actual Close Price", color='blue')
plt.plot(df.index, df['Predicted_Close'], label="Predicted Close Price", color='red', linestyle="dashed")
plt.legend()
plt.title("Stock Price Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
