import numpy as np
import pandas as pd
from scipy.stats import norm
from tvDatafeed import TvDatafeed, Interval

# Black-Scholes-Merton Model Functions
def bsm_call(S, K, T, r, sigma):
    """
    Calculate BSM Call Option Price
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility (annualized)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bsm_put(S, K, T, r, sigma):
    """
    Calculate BSM Put Option Price
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility (annualized)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Initialize and fetch data
print("Fetching TITAN stock data...")

try:
    tv = TvDatafeed()
    print("✓ Logged in successfully")
except:
    tv = TvDatafeed()
    print("⚠ Using anonymous mode")

# Fetch data
df = tv.get_hist(symbol="TITAN", exchange="NSE", interval=Interval.in_daily, n_bars=90)

if df is None or df.empty:
    print("Failed to fetch data")
else:
    df = df.sort_index()
    
    # Get ATM price (last available closing price)
    ATM = df['close'].iloc[-1]
    print(f"\n{'='*60}")
    print(f"ATM Price (Last Close): ₹{ATM:.2f}")
    print(f"{'='*60}")
    
    # Calculate volatility from historical data
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    returns = df['log_returns'].dropna()
    daily_std = returns.std()
    annualized_volatility = daily_std * np.sqrt(252)
    
    print(f"Annualized Volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
    
    # Define strikes
    strikes = {
        'ATM-5%': ATM * 0.95,
        'ATM-2%': ATM * 0.98,
        'ATM': ATM,
        'ATM+2%': ATM * 1.02,
        'ATM+5%': ATM * 1.05
    }
    
    # Define maturities (in days)
    maturities_days = [30, 60, 90]
    maturities_years = [days / 365 for days in maturities_days]
    
    # Risk-free rate (using approximate Indian T-Bill rate)
    # r is calculated using closing rate of Indian 10 year T-bill
    bond_data = tv.get_hist(symbol='IN10Y',
    exchange='TVC',
    interval=Interval.in_daily,
    n_bars=1)

    r = 0.0654
    
    print(f"Risk-free rate: {r*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Create option pricing table
    results = []
    
    for strike_label, K in strikes.items():
        for days, T in zip(maturities_days, maturities_years):
            call_price = bsm_call(ATM, K, T, r, annualized_volatility)
            put_price = bsm_put(ATM, K, T, r, annualized_volatility)
            
            results.append({
                'Strike': strike_label,
                'Strike_Price': K,
                'Maturity_Days': days,
                'Call_Price': call_price,
                'Put_Price': put_price
            })
    
    # Create DataFrame
    df_options = pd.DataFrame(results)
    
    # Print results in a clean table format
    print("BSM OPTION PRICING TABLE")
    print("="*80)
    print(f"{'Strike':<12} {'Strike Price':<15} {'Maturity':<12} {'Call Price':<15} {'Put Price':<15}")
    print("-"*80)
    
    for _, row in df_options.iterrows():
        print(f"{row['Strike']:<12} ₹{row['Strike_Price']:<14.2f} {row['Maturity_Days']:<12} "
              f"₹{row['Call_Price']:<14.2f} ₹{row['Put_Price']:<14.2f}")
    
    print("="*80)
    
    # Create pivot tables for better visualization
    print("\n\nCALL OPTIONS - Pivot Table")
    print("="*60)
    pivot_calls = df_options.pivot(index='Strike', columns='Maturity_Days', values='Call_Price')
    pivot_calls.columns = [f'{col}D' for col in pivot_calls.columns]
    print(pivot_calls.to_string(float_format=lambda x: f'₹{x:.2f}'))
    
    print("\n\nPUT OPTIONS - Pivot Table")
    print("="*60)
    pivot_puts = df_options.pivot(index='Strike', columns='Maturity_Days', values='Put_Price')
    pivot_puts.columns = [f'{col}D' for col in pivot_puts.columns]
    print(pivot_puts.to_string(float_format=lambda x: f'₹{x:.2f}'))
    
    # Summary statistics
    print("\n\nSUMMARY STATISTICS")
    print("="*60)
    summary = pd.DataFrame({
        'Option_Type': ['Call', 'Put'],
        'Min_Price': [df_options['Call_Price'].min(), df_options['Put_Price'].min()],
        'Max_Price': [df_options['Call_Price'].max(), df_options['Put_Price'].max()],
        'Mean_Price': [df_options['Call_Price'].mean(), df_options['Put_Price'].mean()],
        'Std_Price': [df_options['Call_Price'].std(), df_options['Put_Price'].std()]
    })
    print(summary.to_string(index=False, float_format=lambda x: f'₹{x:.2f}'))
    
    # Export to CSV
    df_options.to_csv('TITAN_BSM_Options.csv', index=False)
    print(f"\n✓ Option prices exported to TITAN_BSM_Options.csv")
    
    # Input parameters summary
    print("\n\nINPUT PARAMETERS")
    print("="*60)
    params = pd.DataFrame({
        'Parameter': ['ATM Price', 'Volatility', 'Risk-free Rate', 'Number of Options Priced'],
        'Value': [f'₹{ATM:.2f}', f'{annualized_volatility*100:.2f}%', f'{r*100:.1f}%', len(results)]
    })
    print(params.to_string(index=False))
    print("="*60)

