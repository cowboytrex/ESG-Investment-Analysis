from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START_DATE = "2020-01-01"
END_DATE = "2026-10-13"  # Exclusive end date
TICKERS = {
    "DSI": "ESG Portfolio Proxy",
    "SPY": "S&P 500 Benchmark",
}

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# 1. Download daily adjusted market prices
prices = {}

for ticker, label in TICKERS.items():
    history = yf.Ticker(ticker).history(
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
    )

    if history.empty:
        raise ValueError(f"No price data found for {ticker}")

    prices[label] = history["Close"]

daily_prices = pd.concat(prices, axis=1).dropna()

# 2. Convert daily prices into month-end observations
monthly_prices = daily_prices.resample("ME").last()

# 3. Calculate monthly returns
monthly_returns = monthly_prices.pct_change().dropna()

# 4. Calculate cumulative returns
cumulative_returns = (1 + monthly_returns).cumprod()

# 5. Calculate annualized risk and return
annualized_return = (1 + monthly_returns).prod() ** (
    12 / len(monthly_returns)
) - 1

annualized_volatility = monthly_returns.std() * np.sqrt(12)

# 6. Calculate historical 95% monthly Value at Risk
monthly_var_95 = -monthly_returns.quantile(0.05)

# 7. Calculate maximum drawdown
drawdown = (
    cumulative_returns / cumulative_returns.cummax()
) - 1

maximum_drawdown = drawdown.min()

# 8. Save metrics
summary = pd.DataFrame({
    "Annualized Return": annualized_return,
    "Annualized Volatility": annualized_volatility,
    "Monthly VaR 95%": monthly_var_95,
    "Maximum Drawdown": maximum_drawdown,
})

monthly_prices.to_csv(OUTPUT / "monthly_prices.csv")
monthly_returns.to_csv(OUTPUT / "monthly_returns.csv")
summary.to_csv(OUTPUT / "performance_summary.csv")

print(summary.round(4))

# 9. Visualize cumulative growth
ax = (cumulative_returns * 100).plot(figsize=(11, 6))

ax.set_title("ESG vs S&P 500: Cumulative Growth (2020–2026)")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (Initial = 100)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / "cumulative_returns.png", dpi=200)
plt.show()