import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
from scipy import stats
from datetime import datetime, timedelta


tv = TvDatafeed()

# Extract TITAN stock data for last 3 months
symbol = "TITAN"
exchange = "NSE"
interval = Interval.in_daily
n_bars = 75  # Approximately 3 months of trading days

print(f"Fetching data for {symbol}...")
df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)

if df is None or df.empty:
    print("Failed to fetch data. Please check symbol and exchange.")
else:
    # Sort by date
    df = df.sort_index()
    
    # Compute daily log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Remove NaN values
    returns = df['log_returns'].dropna()
    
    # Compute statistics
    daily_std = returns.std()
    annualized_volatility = daily_std * np.sqrt(252)
    skewness = stats.skew(returns)
    kurtosis_val = stats.kurtosis(returns)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Mean Daily Return',
            'Std Daily Return',
            'Annualized Volatility',
            'Skewness',
            'Kurtosis',
            'Min Return',
            'Max Return',
            'Number of Observations'
        ],
        'Value': [
            f"{returns.mean():.6f}",
            f"{daily_std:.6f}",
            f"{annualized_volatility:.4f}",
            f"{skewness:.4f}",
            f"{kurtosis_val:.4f}",
            f"{returns.min():.6f}",
            f"{returns.max():.6f}",
            len(returns)
        ]
    })
    
    print("\n" + "="*50)
    print(f"SUMMARY STATISTICS - {symbol}")
    print("="*50)
    print(summary_stats.to_string(index=False))
    print("="*50)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{symbol} Stock Analysis - Last 3 Months', fontsize=14, fontweight='bold')
    
    # Plot 1: Stock Price
    axes[0, 0].plot(df.index, df['close'], linewidth=1.5, color='#2E86AB')
    axes[0, 0].set_title('Daily Closing Price', fontsize=11)
    axes[0, 0].set_xlabel('Date', fontsize=9)
    axes[0, 0].set_ylabel('Price (INR)', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
    
    # Plot 2: Daily Log Returns
    axes[0, 1].plot(returns.index, returns, linewidth=1, color='#A23B72', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0, 1].set_title('Daily Log Returns', fontsize=11)
    axes[0, 1].set_xlabel('Date', fontsize=9)
    axes[0, 1].set_ylabel('Log Returns', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
    
    # Plot 3: Distribution of Returns
    axes[1, 0].hist(returns, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
    axes[1, 0].set_title('Distribution of Log Returns', fontsize=11)
    axes[1, 0].set_xlabel('Log Returns', fontsize=9)
    axes[1, 0].set_ylabel('Frequency', fontsize=9)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Q-Q Plot
    stats.probplot(returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional detailed statistics table
    print(f"\nDetailed Statistics:")
    print(returns.describe().to_string())
    
    # Export to CSV (optional)
    df.to_csv(f'{symbol}_data.csv')
    print(f"\nData exported to {symbol}_data.csv")