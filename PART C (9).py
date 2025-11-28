import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        return iv * 100
    except:
        return np.nan

# Read the real option data downloaded from NSE website
df = pd.read_csv('option_chain_data.csv', skiprows=1)
df.columns = df.columns.str.strip()

# Extract data
strikes = pd.to_numeric(df['STRIKE'].str.replace(',', ''), errors='coerce')
call_ltp = pd.to_numeric(df['LTP'], errors='coerce')
put_ltp = pd.to_numeric(df['LTP.1'], errors='coerce')

# Parameters
S = 3900
r = 0.065

# Multiple maturities (in days) - simulating different expiry dates
maturities = [7, 15, 30, 45, 60, 90]

# Build volatility surface data
surface_data = []

for maturity in maturities:
    T = maturity / 365
    
    for i in range(len(strikes)):
        if pd.notna(strikes.iloc[i]):
            K = strikes.iloc[i]
            
            # Calculate IV for calls
            if pd.notna(call_ltp.iloc[i]) and call_ltp.iloc[i] > 0:
                call_iv = calc_iv(call_ltp.iloc[i], S, K, T, r, 'call')
                if pd.notna(call_iv):
                    surface_data.append({
                        'Strike': K,
                        'Maturity': maturity,
                        'Type': 'Call',
                        'IV': call_iv
                    })
            
            # Calculate IV for puts
            if pd.notna(put_ltp.iloc[i]) and put_ltp.iloc[i] > 0:
                put_iv = calc_iv(put_ltp.iloc[i], S, K, T, r, 'put')
                if pd.notna(put_iv):
                    surface_data.append({
                        'Strike': K,
                        'Maturity': maturity,
                        'Type': 'Put',
                        'IV': put_iv
                    })

surface_df = pd.DataFrame(surface_data)

# Create pivot table for surface
pivot_calls = surface_df[surface_df['Type'] == 'Call'].pivot_table(
    values='IV', index='Strike', columns='Maturity', aggfunc='mean'
)
pivot_puts = surface_df[surface_df['Type'] == 'Put'].pivot_table(
    values='IV', index='Strike', columns='Maturity', aggfunc='mean'
)

# Save to Excel with multiple sheets
with pd.ExcelWriter('volatility_surface.xlsx', engine='openpyxl') as writer:
    surface_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    pivot_calls.to_excel(writer, sheet_name='Call_Surface')
    pivot_puts.to_excel(writer, sheet_name='Put_Surface')

print("=" * 60)
print("VOLATILITY SURFACE DATA")
print("=" * 60)
print(f"\nParameters: S={S}, r={r*100:.1f}%")
print(f"Maturities: {maturities} days\n")
print("\nCALL OPTIONS SURFACE:")
print(pivot_calls.round(2))
print("\n\nPUT OPTIONS SURFACE:")
print(pivot_puts.round(2))
print("\n✓ Excel file saved: 'volatility_surface.xlsx'")

# Create 3D surface plot for calls
fig = plt.figure(figsize=(14, 6))

# Plot 1: Call Options
ax1 = fig.add_subplot(121, projection='3d')
X_calls, Y_calls = np.meshgrid(pivot_calls.columns, pivot_calls.index)
Z_calls = pivot_calls.values

surf1 = ax1.plot_surface(X_calls, Y_calls, Z_calls, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Maturity (Days)')
ax1.set_ylabel('Strike Price')
ax1.set_zlabel('Implied Volatility (%)')
ax1.set_title('Volatility Surface - Call Options')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Plot 2: Put Options
ax2 = fig.add_subplot(122, projection='3d')
X_puts, Y_puts = np.meshgrid(pivot_puts.columns, pivot_puts.index)
Z_puts = pivot_puts.values

surf2 = ax2.plot_surface(X_puts, Y_puts, Z_puts, cmap='plasma', alpha=0.8)
ax2.set_xlabel('Maturity (Days)')
ax2.set_ylabel('Strike Price')
ax2.set_zlabel('Implied Volatility (%)')
ax2.set_title('Volatility Surface - Put Options')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('volatility_surface.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: 'volatility_surface.png'")
plt.show()

# Create 2D heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Calls heatmap
im1 = ax1.imshow(pivot_calls.values, cmap='RdYlGn_r', aspect='auto')
ax1.set_xticks(range(len(pivot_calls.columns)))
ax1.set_xticklabels(pivot_calls.columns)
ax1.set_yticks(range(len(pivot_calls.index)))
ax1.set_yticklabels(pivot_calls.index.astype(int))
ax1.set_xlabel('Maturity (Days)')
ax1.set_ylabel('Strike Price')
ax1.set_title('IV Heatmap - Calls')
plt.colorbar(im1, ax=ax1, label='IV (%)')

# Puts heatmap
im2 = ax2.imshow(pivot_puts.values, cmap='RdYlGn_r', aspect='auto')
ax2.set_xticks(range(len(pivot_puts.columns)))
ax2.set_xticklabels(pivot_puts.columns)
ax2.set_yticks(range(len(pivot_puts.index)))
ax2.set_yticklabels(pivot_puts.index.astype(int))
ax2.set_xlabel('Maturity (Days)')
ax2.set_ylabel('Strike Price')
ax2.set_title('IV Heatmap - Puts')
plt.colorbar(im2, ax=ax2, label='IV (%)')

plt.tight_layout()
plt.savefig('volatility_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap saved: 'volatility_heatmap.png'")
plt.show()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total data points: {len(surface_df)}")
print(f"Call options: {len(surface_df[surface_df['Type']=='Call'])}")
print(f"Put options: {len(surface_df[surface_df['Type']=='Put'])}")
print(f"Average IV (Calls): {surface_df[surface_df['Type']=='Call']['IV'].mean():.2f}%")
print(f"Average IV (Puts): {surface_df[surface_df['Type']=='Put']['IV'].mean():.2f}%")

print("=" * 60)
