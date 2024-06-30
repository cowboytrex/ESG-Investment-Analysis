import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from bs4 import BeautifulSoup

# Step 1: Reading ESG Scores from a Local HTML File for Testing
def fetch_esg_data():
    with open('sample_esg.html', 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find the table and extract data
    table = soup.find('table')
    rows = table.find_all('tr')
    
    esg_data = []
    for row in rows[1:]:  # Skip header row
        cols = row.find_all('td')
        company = cols[0].text.strip()
        esg_score = float(cols[1].text.strip())
        esg_data.append([company, esg_score])
    
    return pd.DataFrame(esg_data, columns=['company', 'esg_score'])

# Step 2: Reading Stock Prices from a Local HTML File for Testing
def fetch_stock_data():
    with open('sample_stock.html', 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find the table and extract data
    table = soup.find('table')
    rows = table.find_all('tr')
    
    stock_data = []
    for row in rows[1:]:  # Skip header row
        cols = row.find_all('td')
        company = cols[0].text.strip()
        date = cols[1].text.strip()
        close = float(cols[2].text.strip())
        stock_data.append([company, date, close])
    
    return pd.DataFrame(stock_data, columns=['company', 'date', 'close'])

# Load data
esg_data = fetch_esg_data()
print("ESG Data:\n", esg_data.head())
stock_data = fetch_stock_data()
print("Stock Data:\n", stock_data.head())

# Data cleaning and processing
esg_data.dropna(inplace=True)
stock_data.dropna(inplace=True)
stock_data['date'] = pd.to_datetime(stock_data['date'])

# Normalize ESG scores
esg_data['normalized_score'] = (esg_data['esg_score'] - esg_data['esg_score'].min()) / (esg_data['esg_score'].max() - esg_data['esg_score'].min())

# Select top 50 ESG companies (adjust as necessary based on available data)
top_esg_companies = esg_data.nlargest(50, 'normalized_score')['company']
print("Top ESG Companies:\n", top_esg_companies)

# Filter stock data for ESG portfolio
esg_portfolio = stock_data[stock_data['company'].isin(top_esg_companies)]
print("ESG Portfolio Data:\n", esg_portfolio.head())

# Calculate daily returns
esg_portfolio['return'] = esg_portfolio.groupby('company')['close'].pct_change()
print("ESG Portfolio Returns:\n", esg_portfolio.head())

# Aggregate returns by date
esg_returns = esg_portfolio.groupby('date')['return'].mean()
print("ESG Returns:\n", esg_returns.head())

# Ensure there are no empty DataFrames
if esg_returns.empty:
    print("Error: The ESG return DataFrame is empty.")
    exit(1)

# Calculate cumulative returns
esg_cumulative_returns = (1 + esg_returns.dropna()).cumprod() - 1
print("ESG Cumulative Returns:\n", esg_cumulative_returns.head())

# Calculate performance metrics
esg_mean_return = esg_returns.mean()
esg_volatility = esg_returns.std()

# Calculate Value at Risk (VaR)
confidence_level = 0.95
esg_var = np.percentile(esg_returns.dropna(), (1 - confidence_level) * 100)

# Stress Testing (Example: drop in value in one day)
esg_stress_test = esg_returns.min()

# Visualization of cumulative returns
plt.figure(figsize=(14, 7))
sns.lineplot(data=esg_cumulative_returns, label='ESG Portfolio')
plt.title('Cumulative Returns: ESG Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Visualization of daily returns
plt.figure(figsize=(14, 7))
sns.lineplot(data=esg_returns, label='ESG Portfolio')
plt.title('Daily Returns: ESG Portfolio')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()

# Visualization of Value at Risk
plt.figure(figsize=(14, 7))
sns.histplot(esg_returns.dropna(), kde=True, label='ESG Portfolio', color='blue', stat='density', bins=50)
plt.axvline(x=esg_var, color='blue', linestyle='--', label=f'ESG VaR ({confidence_level*100:.1f}%)')
plt.title('Value at Risk: ESG Portfolio')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.show()

# Summary Report
print("Performance Metrics:")
print(f"ESG Portfolio Mean Daily Return: {esg_mean_return:.4f}")
print(f"ESG Portfolio Volatility: {esg_volatility:.4f}")
print(f"ESG Portfolio VaR (95% confidence): {esg_var:.4f}")
print(f"ESG Portfolio Stress Test (Max Daily Loss): {esg_stress_test:.4f}")
