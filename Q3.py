import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tvDatafeed import TvDatafeed, Interval
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# ==================== BSM GREEKS FUNCTIONS ====================

def bsm_call(S, K, T, r, sigma):
    """BSM Call Price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bsm_put(S, K, T, r, sigma):
    """BSM Put Price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def delta_call(S, K, T, r, sigma):
    """Call Delta"""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    """Put Delta"""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    """Gamma (same for call and put)"""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """Vega (same for call and put)"""
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change

def theta_call(S, K, T, r, sigma):
    """Call Theta"""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    return theta / 365  # Daily theta

def theta_put(S, K, T, r, sigma):
    """Put Theta"""
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return theta / 365  # Daily theta

def rho_call(S, K, T, r, sigma):
    """Call Rho"""
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in rate

def rho_put(S, K, T, r, sigma):
    """Put Rho"""
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change in rate

# ==================== IMPLIED VOLATILITY ====================

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using Newton-Raphson method
    """
    def objective(sigma):
        if option_type == 'call':
            price = bsm_call(S, K, T, r, sigma)
        else:
            price = bsm_put(S, K, T, r, sigma)
        return abs(price - option_price)
    
    try:
        result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
        if result.success and result.fun < 0.01:
            return result.x
        else:
            return np.nan
    except:
        return np.nan

# ==================== FETCH NSE OPTION CHAIN ====================

def fetch_nse_option_chain(symbol="TITAN"):
    """
    Fetch option chain data from NSE
    Note: NSE has restrictions, this is a template
    You may need to use alternative sources or manual data
    """
    print(f"\n⚠ Note: Automated NSE scraping may be restricted.")
    print("For demonstration, generating synthetic market data based on BSM model...")
    
    # For demonstration, we'll create synthetic market prices
    # In practice, you would fetch this from NSE or your broker's API
    return None

# ==================== MAIN ANALYSIS ====================

print("="*80)
print("OPTION GREEKS & IMPLIED VOLATILITY ANALYSIS - TITAN")
print("="*80)

# Initialize and fetch stock data
username = 'your_tradingview_username'
password = 'your_tradingview_password'

try:
    tv = TvDatafeed(username=username, password=password)
    print("✓ Logged in successfully")
except:
    tv = TvDatafeed()
    print("⚠ Using anonymous mode")

# Fetch TITAN data
df = tv.get_hist(symbol="TITAN", exchange="NSE", interval=Interval.in_daily, n_bars=90)

if df is None or df.empty:
    print("Failed to fetch data")
    exit()

df = df.sort_index()

# Get ATM price
ATM = df['close'].iloc[-1]
print(f"\nATM Price (Last Close): ₹{ATM:.2f}")

# Calculate historical volatility
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
returns = df['log_returns'].dropna()
daily_std = returns.std()
hist_vol = daily_std * np.sqrt(252)

print(f"Historical Volatility: {hist_vol:.4f} ({hist_vol*100:.2f}%)")

# Define parameters
strikes = {
    'ATM-5%': ATM * 0.95,
    'ATM-2%': ATM * 0.98,
    'ATM': ATM,
    'ATM+2%': ATM * 1.02,
    'ATM+5%': ATM * 1.05
}

maturities_days = [30, 60, 90]
maturities_years = [days / 365 for days in maturities_days]
r = 0.07  # Risk-free rate

# ==================== CALCULATE GREEKS WITH HISTORICAL VOL ====================

print("\n" + "="*80)
print("GREEKS CALCULATION - HISTORICAL VOLATILITY")
print("="*80)

greeks_hist = []

for strike_label, K in strikes.items():
    for days, T in zip(maturities_days, maturities_years):
        # Calculate option prices
        call_price = bsm_call(ATM, K, T, r, hist_vol)
        put_price = bsm_put(ATM, K, T, r, hist_vol)
        
        # Calculate Greeks
        delta_c = delta_call(ATM, K, T, r, hist_vol)
        delta_p = delta_put(ATM, K, T, r, hist_vol)
        gamma_val = gamma(ATM, K, T, r, hist_vol)
        vega_val = vega(ATM, K, T, r, hist_vol)
        theta_c = theta_call(ATM, K, T, r, hist_vol)
        theta_p = theta_put(ATM, K, T, r, hist_vol)
        rho_c = rho_call(ATM, K, T, r, hist_vol)
        rho_p = rho_put(ATM, K, T, r, hist_vol)
        
        greeks_hist.append({
            'Strike': strike_label,
            'Strike_Price': K,
            'Maturity_Days': days,
            'Vol_Type': 'Historical',
            'Volatility': hist_vol,
            'Call_Price': call_price,
            'Put_Price': put_price,
            'Call_Delta': delta_c,
            'Put_Delta': delta_p,
            'Gamma': gamma_val,
            'Vega': vega_val,
            'Call_Theta': theta_c,
            'Put_Theta': theta_p,
            'Call_Rho': rho_c,
            'Put_Rho': rho_p
        })

df_greeks_hist = pd.DataFrame(greeks_hist)

# ==================== SIMULATE MARKET PRICES & CALCULATE IV ====================

print("\n" + "="*80)
print("IMPLIED VOLATILITY CALCULATION")
print("="*80)
print("Simulating market prices (add noise to theoretical prices)...")

greeks_iv = []

for strike_label, K in strikes.items():
    for days, T in zip(maturities_days, maturities_years):
        # Simulate market prices (add randomness to mimic market conditions)
        theoretical_call = bsm_call(ATM, K, T, r, hist_vol)
        theoretical_put = bsm_put(ATM, K, T, r, hist_vol)
        
        # Add realistic noise (±5-15%)
        noise_factor_call = np.random.uniform(0.90, 1.15)
        noise_factor_put = np.random.uniform(0.90, 1.15)
        
        market_call = theoretical_call * noise_factor_call
        market_put = theoretical_put * noise_factor_put
        
        # Calculate implied volatility
        iv_call = implied_volatility(market_call, ATM, K, T, r, 'call')
        iv_put = implied_volatility(market_put, ATM, K, T, r, 'put')
        
        # Use average IV if both successful
        if not np.isnan(iv_call) and not np.isnan(iv_put):
            iv_avg = (iv_call + iv_put) / 2
        elif not np.isnan(iv_call):
            iv_avg = iv_call
        elif not np.isnan(iv_put):
            iv_avg = iv_put
        else:
            iv_avg = hist_vol  # Fallback to historical
        
        # Recalculate Greeks using IV
        delta_c = delta_call(ATM, K, T, r, iv_avg)
        delta_p = delta_put(ATM, K, T, r, iv_avg)
        gamma_val = gamma(ATM, K, T, r, iv_avg)
        vega_val = vega(ATM, K, T, r, iv_avg)
        theta_c = theta_call(ATM, K, T, r, iv_avg)
        theta_p = theta_put(ATM, K, T, r, iv_avg)
        rho_c = rho_call(ATM, K, T, r, iv_avg)
        rho_p = rho_put(ATM, K, T, r, iv_avg)
        
        greeks_iv.append({
            'Strike': strike_label,
            'Strike_Price': K,
            'Maturity_Days': days,
            'Vol_Type': 'Implied',
            'Volatility': iv_avg,
            'Call_Price': market_call,
            'Put_Price': market_put,
            'Call_Delta': delta_c,
            'Put_Delta': delta_p,
            'Gamma': gamma_val,
            'Vega': vega_val,
            'Call_Theta': theta_c,
            'Put_Theta': theta_p,
            'Call_Rho': rho_c,
            'Put_Rho': rho_p
        })

df_greeks_iv = pd.DataFrame(greeks_iv)

# ==================== COMBINE AND COMPARE ====================

df_greeks_combined = pd.concat([df_greeks_hist, df_greeks_iv], ignore_index=True)

print("\n✓ Greeks calculated for both Historical and Implied volatility")

# ==================== EXPORT TO EXCEL ====================

with pd.ExcelWriter('TITAN_Greeks_IV_Analysis.xlsx', engine='openpyxl') as writer:
    # Full Greeks table
    df_greeks_combined.to_excel(writer, sheet_name='All Greeks', index=False)
    
    # Historical Vol Greeks
    df_greeks_hist.to_excel(writer, sheet_name='Historical Vol Greeks', index=False)
    
    # Implied Vol Greeks
    df_greeks_iv.to_excel(writer, sheet_name='Implied Vol Greeks', index=False)
    
    # Comparison - Delta
    comparison_delta = df_greeks_combined.pivot_table(
        index=['Strike', 'Maturity_Days'],
        columns='Vol_Type',
        values='Call_Delta'
    )
    comparison_delta.to_excel(writer, sheet_name='Delta Comparison')
    
    # Volatility Surface Data
    vol_surface = df_greeks_iv.pivot(
        index='Strike_Price',
        columns='Maturity_Days',
        values='Volatility'
    )
    vol_surface.to_excel(writer, sheet_name='Volatility Surface')

print("✓ Excel file exported: TITAN_Greeks_IV_Analysis.xlsx")

# ==================== PRINT SUMMARY TABLES ====================

print("\n" + "="*80)
print("GREEKS SUMMARY - CALL OPTIONS (Historical Vol)")
print("="*80)
call_summary = df_greeks_hist[['Strike', 'Maturity_Days', 'Call_Price', 
                                'Call_Delta', 'Gamma', 'Vega', 'Call_Theta', 'Call_Rho']]
print(call_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print("\n" + "="*80)
print("GREEKS SUMMARY - PUT OPTIONS (Historical Vol)")
print("="*80)
put_summary = df_greeks_hist[['Strike', 'Maturity_Days', 'Put_Price', 
                               'Put_Delta', 'Gamma', 'Vega', 'Put_Theta', 'Put_Rho']]
print(put_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print("\n" + "="*80)
print("VOLATILITY COMPARISON (Historical vs Implied)")
print("="*80)
vol_comparison = pd.DataFrame({
    'Strike': df_greeks_hist['Strike'],
    'Maturity': df_greeks_hist['Maturity_Days'],
    'Hist_Vol': df_greeks_hist['Volatility'],
    'Impl_Vol': df_greeks_iv['Volatility'],
    'Difference': df_greeks_iv['Volatility'] - df_greeks_hist['Volatility']
})
print(vol_comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ==================== VISUALIZATIONS ====================

fig = plt.figure(figsize=(16, 10))

# 1. Volatility Surface (3D)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
vol_pivot = df_greeks_iv.pivot(index='Strike_Price', columns='Maturity_Days', values='Volatility')
X, Y = np.meshgrid(vol_pivot.columns, vol_pivot.index)
Z = vol_pivot.values
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Maturity (Days)', fontsize=9)
ax1.set_ylabel('Strike Price (₹)', fontsize=9)
ax1.set_zlabel('Implied Volatility', fontsize=9)
ax1.set_title('Implied Volatility Surface', fontsize=11, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 2. Call Delta by Maturity
ax2 = fig.add_subplot(2, 3, 2)
for mat in maturities_days:
    subset = df_greeks_hist[df_greeks_hist['Maturity_Days'] == mat]
    ax2.plot(subset['Strike_Price'], subset['Call_Delta'], marker='o', label=f'{mat}D')
ax2.set_xlabel('Strike Price (₹)', fontsize=9)
ax2.set_ylabel('Call Delta', fontsize=9)
ax2.set_title('Call Delta across Strikes', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# 3. Gamma Profile
ax3 = fig.add_subplot(2, 3, 3)
for mat in maturities_days:
    subset = df_greeks_hist[df_greeks_hist['Maturity_Days'] == mat]
    ax3.plot(subset['Strike_Price'], subset['Gamma'], marker='s', label=f'{mat}D')
ax3.set_xlabel('Strike Price (₹)', fontsize=9)
ax3.set_ylabel('Gamma', fontsize=9)
ax3.set_title('Gamma across Strikes', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Vega Profile
ax4 = fig.add_subplot(2, 3, 4)
for mat in maturities_days:
    subset = df_greeks_hist[df_greeks_hist['Maturity_Days'] == mat]
    ax4.plot(subset['Strike_Price'], subset['Vega'], marker='^', label=f'{mat}D')
ax4.set_xlabel('Strike Price (₹)', fontsize=9)
ax4.set_ylabel('Vega', fontsize=9)
ax4.set_title('Vega across Strikes', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Theta (Call) Profile
ax5 = fig.add_subplot(2, 3, 5)
for mat in maturities_days:
    subset = df_greeks_hist[df_greeks_hist['Maturity_Days'] == mat]
    ax5.plot(subset['Strike_Price'], subset['Call_Theta'], marker='d', label=f'{mat}D')
ax5.set_xlabel('Strike Price (₹)', fontsize=9)
ax5.set_ylabel('Call Theta (Daily)', fontsize=9)
ax5.set_title('Call Theta across Strikes', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Historical vs Implied Vol Comparison
ax6 = fig.add_subplot(2, 3, 6)
strike_labels = df_greeks_hist['Strike'].unique()
x = np.arange(len(strike_labels))
width = 0.35
hist_vols = df_greeks_hist.groupby('Strike')['Volatility'].mean()
impl_vols = df_greeks_iv.groupby('Strike')['Volatility'].mean()
ax6.bar(x - width/2, hist_vols, width, label='Historical', color='#2E86AB')
ax6.bar(x + width/2, impl_vols, width, label='Implied', color='#F18F01')
ax6.set_xlabel('Strike', fontsize=9)
ax6.set_ylabel('Volatility', fontsize=9)
ax6.set_title('Historical vs Implied Volatility', fontsize=11, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(strike_labels, rotation=45, fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('TITAN_Greeks_IV_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: TITAN_Greeks_IV_Analysis.png")
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("Files generated:")
print("  1. TITAN_Greeks_IV_Analysis.xlsx - Complete Greeks and IV data")
print("  2. TITAN_Greeks_IV_Analysis.png - Visualization plots")
print("="*80)