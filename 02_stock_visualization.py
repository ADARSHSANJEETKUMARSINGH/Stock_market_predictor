import yfinance as yf # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Stock Symbol (Apple - AAPL)
stock_symbol = "AAPL"

# Stock Data Fetch
stock = yf.Ticker(stock_symbol)
stock_data = stock.history(period="1mo")  # 1 month ka data

# Graph Banayein
plt.figure(figsize=(10, 5))
plt.plot(stock_data.index, stock_data["Close"], marker="o", linestyle="-", color="b", label="Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title(f"{stock_symbol} Stock Closing Price (Last 1 Month)")
plt.legend()
plt.xticks(rotation=45)  # Dates ko thoda tilt karne ke liye
plt.grid()

# Graph Show Karna
plt.show()
