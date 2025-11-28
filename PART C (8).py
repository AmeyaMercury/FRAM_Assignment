import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def calc_iv(market_price, S, K, T, r, option_type='call'):
    try:
        iv = brentq(lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price, 0.01, 5)
        return iv * 100  # Convert to percentage
    except:
        return np.nan

def calc_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100  # Per 1% change
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return delta, gamma, vega, theta, rho

# Read CSV
df = pd.read_csv('option_chain_data.csv', skiprows=1)
df.columns = df.columns.str.strip()

# Extract data
strikes = pd.to_numeric(df['STRIKE'].str.replace(',', ''), errors='coerce')
call_ltp = pd.to_numeric(df['LTP'], errors='coerce')
put_ltp = pd.to_numeric(df['LTP.1'], errors='coerce')

# Parameters (adjust these)
S = 3900  # Current stock price (estimate from ATM strikes)
r = 0.065  # Risk-free rate (6.5%)
T = 30/365  # Time to expiry (30 days assumption)

# Historical volatility (you can adjust this based on actual data)
historical_vol = 0.2522  # 25.22% annualized

results = []
for i in range(len(strikes)):
    if pd.notna(strikes.iloc[i]):
        K = strikes.iloc[i]
        
        # Calculate IV for calls
        if pd.notna(call_ltp.iloc[i]) and call_ltp.iloc[i] > 0:
            call_iv = calc_iv(call_ltp.iloc[i], S, K, T, r, 'call')
            if pd.notna(call_iv):
                # Greeks with calculated IV
                delta_iv, gamma_iv, vega_iv, theta_iv, rho_iv = calc_greeks(S, K, T, r, call_iv/100, 'call')
                # Greeks with historical volatility
                delta_hv, gamma_hv, vega_hv, theta_hv, rho_hv = calc_greeks(S, K, T, r, historical_vol, 'call')
                
                results.append({
                    'Strike': K, 
                    'Type': 'Call', 
                    'LTP': call_ltp.iloc[i], 
                    'IV': call_iv,
                    'HV': historical_vol * 100,
                    'Delta_IV': delta_iv,
                    'Delta_HV': delta_hv,
                    'Gamma_IV': gamma_iv,
                    'Gamma_HV': gamma_hv,
                    'Vega_IV': vega_iv,
                    'Vega_HV': vega_hv,
                    'Theta_IV': theta_iv,
                    'Theta_HV': theta_hv,
                    'Rho_IV': rho_iv,
                    'Rho_HV': rho_hv
                })
        
        # Calculate IV for puts
        if pd.notna(put_ltp.iloc[i]) and put_ltp.iloc[i] > 0:
            put_iv = calc_iv(put_ltp.iloc[i], S, K, T, r, 'put')
            if pd.notna(put_iv):
                # Greeks with calculated IV
                delta_iv, gamma_iv, vega_iv, theta_iv, rho_iv = calc_greeks(S, K, T, r, put_iv/100, 'put')
                # Greeks with historical volatility
                delta_hv, gamma_hv, vega_hv, theta_hv, rho_hv = calc_greeks(S, K, T, r, historical_vol, 'put')
                
                results.append({
                    'Strike': K, 
                    'Type': 'Put', 
                    'LTP': put_ltp.iloc[i], 
                    'IV': put_iv,
                    'HV': historical_vol * 100,
                    'Delta_IV': delta_iv,
                    'Delta_HV': delta_hv,
                    'Gamma_IV': gamma_iv,
                    'Gamma_HV': gamma_hv,
                    'Vega_IV': vega_iv,
                    'Vega_HV': vega_hv,
                    'Theta_IV': theta_iv,
                    'Theta_HV': theta_hv,
                    'Rho_IV': rho_iv,
                    'Rho_HV': rho_hv
                })

results_df = pd.DataFrame(results)
results_df = results_df[results_df['IV'].notna()]

print("=" * 80)
print("IMPLIED VOLATILITY & GREEKS COMPARISON (IV vs HV)")
print("=" * 80)
print(f"\nParameters: S={S}, r={r*100:.1f}%, T={T*365:.0f} days")
print(f"Historical Volatility: {historical_vol*100:.1f}%\n")

# Format display
display_df = results_df.copy()
for col in ['Delta_IV', 'Delta_HV', 'Vega_IV', 'Vega_HV', 'Theta_IV', 'Theta_HV', 'Rho_IV', 'Rho_HV']:
    display_df[col] = display_df[col].round(4)
for col in ['Gamma_IV', 'Gamma_HV']:
    display_df[col] = display_df[col].round(6)
display_df['IV'] = display_df['IV'].round(2)
display_df['HV'] = display_df['HV'].round(2)

print(display_df.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
calls = results_df[results_df['Type'] == 'Call']
puts = results_df[results_df['Type'] == 'Put']

if len(calls) > 0:
    print(f"\nCALLS:")
    print(f"  Avg IV: {calls['IV'].mean():.2f}%")
    print(f"  Avg Delta Diff (IV-HV): {(calls['Delta_IV'] - calls['Delta_HV']).mean():.4f}")
    print(f"  Avg Gamma Diff (IV-HV): {(calls['Gamma_IV'] - calls['Gamma_HV']).mean():.6f}")
    
if len(puts) > 0:
    print(f"\nPUTS:")
    print(f"  Avg IV: {puts['IV'].mean():.2f}%")
    print(f"  Avg Delta Diff (IV-HV): {(puts['Delta_IV'] - puts['Delta_HV']).mean():.4f}")
    print(f"  Avg Gamma Diff (IV-HV): {(puts['Gamma_IV'] - puts['Gamma_HV']).mean():.6f}")

print(f"\nOVERALL:")
print(f"  Avg IV: {results_df['IV'].mean():.2f}%")
print(f"  HV: {historical_vol*100:.1f}%")
print(f"  IV-HV Spread: {results_df['IV'].mean() - historical_vol*100:.2f}%")
print("=" * 80)

# Save to CSV
results_df.to_csv('greeks_comparison.csv', index=False)
print("\n✓ Results saved to 'greeks_comparison.csv'")