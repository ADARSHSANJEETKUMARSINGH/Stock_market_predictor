import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# 📌 Stock Data Fetch Karna
ticker = "AAPL"  # Apple stock
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# 📌 Extra Features Add Karna
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['Volatility'] = df['High'] - df['Low']  # Daily Volatility

# 📌 Relative Strength Index (RSI) Calculate Karna
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 📌 Target Variable (Tomorrow's Close Price)
df['Tomorrow'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# 📌 Feature Selection (Current Close Feature Hata Diya)
X = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'EMA_10', 'Volatility', 'RSI']]
y = df['Tomorrow']

# 📌 Train-Test Split (Overfitting ko avoid karne ke liye 70-30 kiya)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 📌 Feature Scaling (Normalize Karna)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Machine Learning Model (Overfitting Avoid Karne Ke Liye Tune Kiya)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 📌 Model Ka Prediction Lena
predictions = model.predict(X_test)

# 📌 Model Ki Accuracy Check Karna (R² Score)
accuracy = model.score(X_test, y_test)
print(f"Final Model Accuracy: {accuracy:.2f}")  # Accuracy Print Karega

# 📌 Model Save Karna
joblib.dump(model, "stock_price_predictor.pkl")
print("Model saved successfully!")

# 📌 Graph Plot Karna (Overfitting fix karne ke liye smoothing add kiya)
predictions_smooth = pd.Series(predictions).rolling(window=5).mean()
y_test_smooth = pd.Series(y_test).rolling(window=5).mean()

plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], y_test_smooth, label="Actual Close Price", color="blue")
plt.plot(df.index[-len(y_test):], predictions_smooth, label="Predicted Close Price", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Prediction vs Actual")
plt.legend()
plt.show()
# 📌 Model Save Karna
joblib.dump(model, "stock_price_predictor.pkl")  # Model Save
joblib.dump(scaler, "scaler.pkl")  # Scaler Save
print("Model & Scaler saved successfully!")
