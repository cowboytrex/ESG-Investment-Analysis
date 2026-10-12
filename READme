# ESG Investment Analysis & Canadian Bank Stock Performance

A Python-based financial analytics project exploring Environmental, Social, and Governance (ESG) investing, portfolio performance evaluation, and historical stock market trends.

This repository combines two financial analysis workflows: an ESG-focused investment analysis and a historical stock performance comparison of Canada's six major banks.

## Project Overview

This project demonstrates how Python can be used to collect, process, analyze, and visualize financial datasets to better understand investment performance and financial risk.

The project consists of two components:

**1. ESG Investment Analysis**

An exploratory analysis examining how ESG scores can be incorporated into portfolio selection and investment performance evaluation.

**2. Canadian Bank Stock Performance Analysis (2020–2026)**

A historical comparison of six major Canadian banks using real market data retrieved through Yahoo Finance. The analysis examines monthly returns and normalized stock performance.

## Key Features

### ESG Investment Analysis

- **Data Extraction:** Extract ESG scores and stock price data from local HTML files using BeautifulSoup.
- **ESG Ranking:** Normalize ESG scores and rank companies based on ESG-related metrics.
- **Portfolio Construction:** Select companies according to ESG rankings.
- **Return Analysis:** Calculate periodic and cumulative investment returns.
- **Risk Evaluation:** Explore volatility, historical Value at Risk (VaR), and worst observed period returns.
- **Data Visualization:** Generate charts illustrating portfolio performance and return distributions.

### Canadian Bank Stock Performance Analysis

- **Historical Market Data:** Retrieve adjusted historical stock prices using yfinance.
- **Data Processing:** Clean financial datasets and convert daily observations into monthly data.
- **Performance Comparison:** Normalize stock prices to a common starting value of 100.
- **Return Calculation:** Calculate monthly percentage returns.
- **Financial Visualization:** Compare historical stock performance using Matplotlib.
- **Automated Analysis:** Download, process, and visualize financial data through a reproducible Python workflow.

## Canadian Banks Analyzed

| Bank | Stock Ticker |
|---|---|
| Royal Bank of Canada (RBC) | RY.TO |
| Toronto-Dominion Bank (TD) | TD.TO |
| Bank of Nova Scotia (Scotiabank) | BNS.TO |
| Bank of Montreal (BMO) | BMO.TO |
| Canadian Imperial Bank of Commerce (CIBC) | CM.TO |
| National Bank of Canada | NA.TO |

The historical analysis covers January 2020 through the latest available observations in October 2026.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Financial data processing and analysis |
| NumPy | Numerical computations |
| BeautifulSoup | HTML data extraction |
| yfinance | Historical financial market data retrieval |
| Matplotlib | Financial data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical analysis |

## Project Structure

```text
ESG-Investment-Analysis/
├── esg_investment_analysis.py
├── bank_analysis.py
├── sample_stock.html
├── stock_prices.csv
├── outputs/
│   └── bank_performance.png
├── README.md
└── .gitignore
```

Additional sample data files and generated CSV outputs may be present depending on the analysis performed.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/cowboytrex/ESG-Investment-Analysis.git
cd ESG-Investment-Analysis
```

### 2. Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scipy beautifulsoup4 yfinance
```

## How to Run

### Run the ESG Investment Analysis

```bash
python esg_investment_analysis.py
```

This script processes the sample ESG and financial data, ranks companies, constructs a sample portfolio, and evaluates investment performance and selected risk metrics.

Ensure the required sample files are available in the locations expected by the script.

### Run the Canadian Bank Stock Performance Analysis

```bash
python bank_analysis.py
```

The script will:

1. Download historical adjusted closing prices for six Canadian banks.
2. Process and organize the daily stock price data.
3. Aggregate prices into month-end observations.
4. Calculate monthly stock returns.
5. Normalize prices to a common starting value of 100.
6. Generate a comparative stock performance chart.

The visualization is saved to:

`outputs/bank_performance.png`

## Canadian Bank Stock Performance Visualization

![Canadian Bank Stock Performance](outputs/bank_performance.png)

The chart illustrates the relative historical performance of Canada's six major banks between 2020 and 2026.

Stock prices are normalized to a starting value of 100, allowing comparisons across companies with different initial share prices.

A normalized value of 150 represents a 50% increase relative to the starting observation.

The chart is based on historical adjusted stock prices and does not represent a prediction of future performance.

## Methodology

### ESG Investment Analysis

The ESG analysis follows six primary steps:

1. Extract sample ESG scores and historical stock prices.
2. Clean and normalize the datasets.
3. Rank companies according to ESG scores.
4. Construct a sample portfolio using ESG-based selection criteria.
5. Calculate periodic returns and exploratory risk metrics.
6. Visualize investment performance and risk characteristics.

### Canadian Bank Stock Performance Analysis

The Canadian bank analysis follows a historical time-series workflow:

1. Retrieve daily market data using yfinance.
2. Extract adjusted closing prices for each bank.
3. Resample daily data into month-end prices.
4. Calculate monthly percentage returns.
5. Normalize prices relative to the first available month-end observation.
6. Compare historical performance through visualization.

**Normalized Stock Performance**

Normalized Price = (Current Adjusted Price / Initial Adjusted Price) × 100

**Monthly Return**

Monthly Return = (Current Month Price / Previous Month Price) − 1

These calculations provide a consistent framework for comparing stock performance over time.

## Limitations

- The ESG component uses illustrative datasets rather than comprehensive verified ESG ratings.
- The Canadian bank component uses historical stock prices and does not forecast future performance.
- The Canadian bank analysis does not currently incorporate ESG ratings.
- Historical returns do not guarantee future investment results.
- Data quality and availability depend on the external market data provider.
- Risk and return metrics must be interpreted according to the frequency of the underlying observations.

## Future Improvements

- **ESG Data Integration:** Incorporate larger, verified ESG datasets alongside real financial market data.
- **Benchmark Comparison:** Compare ESG-focused portfolios against traditional market benchmarks.
- **Risk-Adjusted Performance:** Add Sharpe ratio, maximum drawdown, and other financial risk metrics.
- **Portfolio Optimization:** Implement configurable portfolio weights and allocation strategies.
- **Expanded Market Coverage:** Analyze additional sectors, financial institutions, and geographic markets.
- **Interactive Dashboard:** Develop an interactive dashboard for exploring financial performance.
- **Automated Testing:** Add data validation, unit tests, and reproducible analysis pipelines.

## Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes only.

The analyses and visualizations do not constitute financial or investment advice.
