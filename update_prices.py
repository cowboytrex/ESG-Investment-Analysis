import yfinance as yf
from pathlib import Path

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]

output = Path("data")
output.mkdir(exist_ok=True)

for ticker in tickers:
    data = yf.Ticker(ticker).history(
        start="2023-01-01",
        interval="1mo",
        auto_adjust=True
    )

    if data.empty:
        print(f"No data found for {ticker}")
        continue

    data.to_csv(output / f"{ticker}_prices.csv")
    
    print(f"Updated {ticker}: {len(data)} observations")