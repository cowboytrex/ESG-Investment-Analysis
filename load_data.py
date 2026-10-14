
import sqlite3
import yfinance as yf
import pandas as pd

from database import create_database, DATABASE

# Six major Canadian banks
banks = [
    "RY.TO",
    "TD.TO",
    "BNS.TO",
    "BMO.TO",
    "CM.TO",
    "NA.TO"
]


def load_data():
    create_database()

    conn = sqlite3.connect(DATABASE)

    try:
        for ticker in banks:
            print(f"Downloading {ticker}...")

            df = yf.download(
                ticker,
                start="2020-01-01",
                end="2026-10-01",
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                print(f"No data found for {ticker}")
                continue

            # Select closing prices
            close = df["Close"]

            # Handle different yfinance column formats
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            records = []

            for date, price in close.dropna().items():
                records.append((
                    ticker,
                    date.strftime("%Y-%m-%d"),
                    float(price)
                ))

            # Save prices into SQL
            conn.executemany("""
                INSERT OR IGNORE INTO stock_prices
                (ticker, date, closing_price)
                VALUES (?, ?, ?)
            """, records)

            conn.commit()
            print(f"Processed {len(records)} prices for {ticker}")

    finally:
        conn.close()


if __name__ == "__main__":
    load_data()
    print("Finished loading stock data!")
