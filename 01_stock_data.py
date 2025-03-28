import yfinance as yf #type: ignore

# Stock Symbol (Example: Tesla - TSLA, Apple - AAPL, Google - GOOGL)
stock_symbol = "AAPL"  # Apple ka stock fetch karenge

# Stock Data Fetch Karna
stock = yf.Ticker(stock_symbol)

# Last 5 Days ka Data
stock_data = stock.history(period="5d")

print(stock_data)
