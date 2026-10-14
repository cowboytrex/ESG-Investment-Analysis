from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Six major Canadian banks
banks = {
    "RY.TO": "RBC",
    "TD.TO": "TD",
    "BNS.TO": "Scotiabank",
    "BMO.TO": "BMO",
    "CM.TO": "CIBC",
    "NA.TO": "National Bank",
}

# Create output folder
output = Path("outputs")
output.mkdir(exist_ok=True)

prices = {}

# Download historical stock prices
for ticker, name in banks.items():
    data = yf.Ticker(ticker).history(
        start="2020-01-01",
        end="2026-10-13",
        auto_adjust=True
    )

    if data.empty:
        print(f"No data found for {name}")
        continue

    prices[name] = data["Close"]
    print(f"Downloaded data for {name}")

if len(prices) != len(banks):
    raise RuntimeError("Some bank price histories could not be downloaded.")

# Combine all banks into one table
daily_prices = pd.concat(prices, axis=1).dropna()

# Convert to monthly prices
monthly_prices = daily_prices.resample("ME").last()

# Calculate monthly returns
monthly_returns = monthly_prices.pct_change().dropna()

# Normalize starting value to 100
normalized = monthly_prices / monthly_prices.iloc[0] * 100

# Save results
monthly_prices.to_csv(output / "bank_monthly_prices.csv")
monthly_returns.to_csv(output / "bank_monthly_returns.csv")

# Create chart

import matplotlib.ticker as mticker

# Professional financial report styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

colors = {
    "RBC": "#2563EB",
    "TD": "#16A34A",
    "Scotiabank": "#EF4444",
    "BMO": "#0EA5E9",
    "CIBC": "#9333EA",
    "National Bank": "#F59E0B",
}

fig, ax = plt.subplots(figsize=(14, 7))

fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

for bank in normalized.columns:
    ax.plot(
        normalized.index,
        normalized[bank],
        label=bank,
        color=colors[bank],
        linewidth=2.5,
    )


ax.text(
    0, 1.035,
    "2020–2026  |  Growth of a $100 investment  |  CAD",
    transform=ax.transAxes,
    fontsize=11,
    color="#64748B",
)

ax.set_ylabel("Investment Value (CAD)", labelpad=15)
ax.set_xlabel("")

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("${x:,.0f}")
)

ax.grid(axis="y", alpha=0.15)
ax.grid(axis="x", visible=False)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0, 1.01),
    ncol=3,
    frameon=False,
    fontsize=10,
    columnspacing=2.5,
)

ax.set_title(
    "Canadian Bank Stock Performance",
    fontsize=20,
    fontweight="bold",
    loc="left",
    pad=75,
)

ax.margins(x=0.01)

fig.text(
    0.125, 0.015,
    "Source: Yahoo Finance via yfinance | Adjusted prices | "
    "Monthly observations | 2020–2026",
    fontsize=9,
    color="#64748B",
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.16, top=0.78)
plt.savefig(
    "outputs/bank_performance.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.show()