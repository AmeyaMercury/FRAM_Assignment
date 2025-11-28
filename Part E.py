import os
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

OPT_CSV = "TITAN_BSM_Options.csv"
LAST60_CSV = "last60daytitan.csv"
OUT_DIR = "."
RISK_FREE_RATE = 0.06513
MC_DRAWS = 100_000
INCLUDE_VEGA_IN_VAR = True  
USE_FULL_REPRICING = True   
GREEK_TOLERANCE = 1e-10     

# ---------- Black-Scholes helpers ----------
def bs_price_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or math.isnan(sigma):
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def bs_price_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or math.isnan(sigma):
        return max(K - S, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol_call(market_price, S, K, T, r, lower=1e-6, upper=5.0):
    intrinsic = max(S - K, 0.0)
    if market_price < intrinsic - 1e-9:
        return np.nan
    try:
        f = lambda sigma: bs_price_call(S, K, T, r, sigma) - market_price
        return brentq(f, lower, upper, maxiter=200, xtol=1e-6)
    except Exception:
        return np.nan

def greeks_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or math.isnan(sigma):
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)  # per 1% vol change (divide by 100 for per 1 vol point)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)
    rho = K * T * math.exp(-r*T) * norm.cdf(d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def greeks_put_from_call(S, K, T, r, sigma):
    c = greeks_call(S, K, T, r, sigma)
    return dict(delta=c['delta'] - 1.0, gamma=c['gamma'], vega=c['vega'],
                theta=c['theta'] + r*K*math.exp(-r*T), rho=c['rho'] - K*T*math.exp(-r*T))

# ---------- Utility ----------
def detect_price_column(df):
    candidates = ["Close","close","Adj Close","Adj_Close","AdjClose","ClosePrice","PRICE","Last"]
    for c in candidates:
        if c in df.columns:
            return c
    if df.shape[1] >= 2:
        return df.columns[1]
    raise ValueError("Could not detect price column in last60 CSV.")

# ---------- NEW: Full portfolio repricing function ----------
def reprice_portfolio(portfolio_df, greeks_df, S_new, vol_shock=0.0):
    """
    Reprice entire portfolio at new spot price and optional vol shock.
    Returns new portfolio value.
    """
    pv = 0.0
    for _, row in portfolio_df.iterrows():
        K = row['K']
        T_days = row['T_days']
        qty = row['Qty']
        instr_type = row['Type']
        
        # Find IV for this instrument
        match = greeks_df[(greeks_df['K']==K) & (greeks_df['T_days']==T_days)]
        if match.empty:
            pv += qty * row['Price']  # fallback
            continue
            
        iv = match.iloc[0]['ImpliedVol']
        if np.isnan(iv):
            pv += qty * row['Price']
            continue
        
        # Apply vol shock if any
        iv_shocked = iv + vol_shock
        T = T_days / 365.0
        
        if instr_type == 'Call':
            new_price = bs_price_call(S_new, K, T, RISK_FREE_RATE, iv_shocked)
        else:
            new_price = bs_price_put(S_new, K, T, RISK_FREE_RATE, iv_shocked)
        
        pv += qty * new_price
    
    return pv

# ---------- Load data ----------
opts = pd.read_csv(OPT_CSV)
last60 = pd.read_csv(LAST60_CSV, parse_dates=[0])

price_col = detect_price_column(last60)
last60 = last60.sort_values(last60.columns[0]).reset_index(drop=True)
last60['simple_return'] = last60[price_col].pct_change()
last60['log_return'] = np.log(last60[price_col] / last60[price_col].shift(1))
returns_simple = last60['simple_return'].dropna().values
returns_log = last60['log_return'].dropna().values

if len(returns_simple) < 30:
    raise ValueError("Not enough return points in last60 CSV.")

S_spot = float(last60[price_col].iloc[-1])

# Estimate historical volatility for vega
hist_vol_simple = np.std(returns_simple, ddof=1) * np.sqrt(252)
hist_vol_log = np.std(returns_log, ddof=1) * np.sqrt(252)

# ---------- Compute IV and Greeks ----------
atm_rows = opts[opts['Strike'].astype(str).str.upper().str.contains('ATM')]
if atm_rows.shape[0] == 0:
    S_for_iv = float(opts['Strike_Price'].iloc[0])
else:
    S_for_iv = float(atm_rows.iloc[0]['Strike_Price'])

greeks_rows = []
for _, r in opts.iterrows():
    K = float(r['Strike_Price'])
    T = float(r['Maturity_Days']) / 365.0
    call_mkt = float(r['Call_Price'])
    put_mkt = float(r['Put_Price'])
    iv = implied_vol_call(call_mkt, S_for_iv, K, T, RISK_FREE_RATE)
    c_g = greeks_call(S_for_iv, K, T, RISK_FREE_RATE, iv)
    p_g = greeks_put_from_call(S_for_iv, K, T, RISK_FREE_RATE, iv)
    greeks_rows.append({
        'StrikeLabel': r['Strike'], 'K': K, 'T_days': int(r['Maturity_Days']),
        'Call_Price': call_mkt, 'Put_Price': put_mkt, 'ImpliedVol': iv,
        'Call_delta': c_g['delta'], 'Call_gamma': c_g['gamma'], 'Call_vega': c_g['vega'],
        'Put_delta': p_g['delta'], 'Put_gamma': p_g['gamma'], 'Put_vega': p_g['vega']
    })

greeks_df = pd.DataFrame(greeks_rows)

# ---------- Build portfolio ----------
atm_df = greeks_df[greeks_df['StrikeLabel'].astype(str).str.upper().str.contains('ATM')].reset_index(drop=True)
if atm_df.shape[0] == 0:
    raise ValueError("No ATM rows found.")

positions = []
for _, r in atm_df.iterrows():
    positions.append({
        'Instrument': f"Call_{r['StrikeLabel']}_{r['T_days']}d",
        'Type': 'Call', 'K': r['K'], 'T_days': r['T_days'], 'Qty': 1,
        'Price': r['Call_Price'], 'Delta': r['Call_delta'], 'Gamma': r['Call_gamma'], 'Vega': r['Call_vega']
    })
    positions.append({
        'Instrument': f"Put_{r['StrikeLabel']}_{r['T_days']}d",
        'Type': 'Put', 'K': r['K'], 'T_days': r['T_days'], 'Qty': -1,
        'Price': r['Put_Price'], 'Delta': r['Put_delta'], 'Gamma': r['Put_gamma'], 'Vega': r['Put_vega']
    })

port_df = pd.DataFrame(positions)
port_df['Delta_contrib'] = port_df['Qty'] * port_df['Delta']
port_df['Gamma_contrib'] = port_df['Qty'] * port_df['Gamma']
port_df['Vega_contrib']  = port_df['Qty'] * port_df['Vega']

total_delta = port_df['Delta_contrib'].sum()
total_gamma = port_df['Gamma_contrib'].sum()
total_vega = port_df['Vega_contrib'].sum()
initial_port_value = (port_df['Qty'] * port_df['Price']).sum()

# ---------- Hedging ----------
stock_qty_delta = -total_delta

call30_rows = port_df[(port_df['Type']=='Call') & (port_df['T_days']==30)]
if call30_rows.shape[0] == 0:
    call30_row = port_df[port_df['Type']=='Call'].iloc[0]
else:
    call30_row = call30_rows.iloc[0]

call30_gamma = call30_row['Gamma']
call30_delta = call30_row['Delta']
call30_vega = call30_row['Vega']
call30_price = call30_row['Price']

n_gamma_calls = 0.0 if (abs(call30_gamma) < GREEK_TOLERANCE) else -total_gamma / call30_gamma
delta_after_gamma = total_delta + n_gamma_calls * call30_delta
vega_after_gamma = total_vega + n_gamma_calls * call30_vega
stock_qty_after = -delta_after_gamma

# ---------- NEW: Create hedge position DataFrames ----------
# Delta hedge position
delta_hedge_pos = pd.DataFrame([{
    'Instrument': 'Stock_Delta_Hedge',
    'Type': 'Stock',
    'K': 0,
    'T_days': 0,
    'Qty': stock_qty_delta,
    'Price': S_spot,
    'Delta': 1.0,
    'Gamma': 0.0,
    'Vega': 0.0
}])

# Gamma hedge positions
gamma_hedge_pos = pd.DataFrame([{
    'Instrument': f'Call_{call30_row["Instrument"]}_GammaHedge',
    'Type': 'Call',
    'K': call30_row['K'],
    'T_days': call30_row['T_days'],
    'Qty': n_gamma_calls,
    'Price': call30_price,
    'Delta': call30_delta,
    'Gamma': call30_gamma,
    'Vega': call30_vega
}, {
    'Instrument': 'Stock_Final_DeltaHedge',
    'Type': 'Stock',
    'K': 0,
    'T_days': 0,
    'Qty': stock_qty_after,
    'Price': S_spot,
    'Delta': 1.0,
    'Gamma': 0.0,
    'Vega': 0.0
}])

# Combined portfolios
port_delta_hedged = pd.concat([port_df, delta_hedge_pos], ignore_index=True)
port_gamma_hedged = pd.concat([port_df, gamma_hedge_pos], ignore_index=True)

# ---------- VaR Calculation Functions ----------
def pnl_quadratic(delta, gamma, vega, returns, vol_changes, S):
    """Quadratic approximation with optional vega"""
    dS = S * returns
    pnl = delta * dS + 0.5 * gamma * (dS**2)
    if INCLUDE_VEGA_IN_VAR and vega != 0:
        pnl += vega * vol_changes / 100  # vega is per 1% vol change
    return pnl

def historical_var_full_repricing(portfolio_df, greeks_df, S_spot, returns, vol_changes):
    """
    Historical VaR using full repricing instead of Greek approximation.
    This captures all nonlinear effects properly.
    """
    initial_val = reprice_portfolio(portfolio_df, greeks_df, S_spot, vol_shock=0.0)
    
    pnls = []
    for i, ret in enumerate(returns):
        S_new = S_spot * (1 + ret)
        vol_shock = vol_changes[i] if INCLUDE_VEGA_IN_VAR else 0.0
        
        # Reprice entire portfolio
        new_val = reprice_portfolio(portfolio_df, greeks_df, S_new, vol_shock=vol_shock)
        
        # Add stock hedge P&L if present
        stock_pnl = 0.0
        for _, row in portfolio_df.iterrows():
            if row['Type'] == 'Stock':
                stock_pnl += row['Qty'] * (S_new - S_spot)
        
        pnl = (new_val - initial_val) + stock_pnl
        pnls.append(pnl)
    
    pnls = np.array(pnls)
    var95 = -np.percentile(pnls, 5)
    var99 = -np.percentile(pnls, 1)
    return var95, var99, pnls

# Estimate historical vol changes (for vega risk)
if INCLUDE_VEGA_IN_VAR:
    # Rolling 20-day vol
    rolling_vol = last60[price_col].pct_change().rolling(20).std() * np.sqrt(252)
    vol_changes = rolling_vol.diff().dropna().values
    # Align with returns
    if len(vol_changes) < len(returns_simple):
        vol_changes = np.pad(vol_changes, (0, len(returns_simple) - len(vol_changes)), mode='constant', constant_values=0)
    vol_changes = vol_changes[-len(returns_simple):]
else:
    vol_changes = np.zeros_like(returns_simple)

sigma_simple = np.std(returns_simple, ddof=1)
sigma_log = np.std(returns_log, ddof=1)
z95, z99 = norm.ppf(0.95), norm.ppf(0.99)

# ---------- Compute Greeks for all portfolios ----------
portfolios = {
    'Unhedged': (total_delta, total_gamma, total_vega, port_df),
    'Delta-hedged': (total_delta + stock_qty_delta, total_gamma, total_vega, port_delta_hedged),
    'Gamma-hedged': (delta_after_gamma, total_gamma + n_gamma_calls * call30_gamma, 
                     vega_after_gamma, port_gamma_hedged),
    'Delta+Gamma-hedged': (delta_after_gamma + stock_qty_after, 
                           total_gamma + n_gamma_calls * call30_gamma,
                           vega_after_gamma, port_gamma_hedged)
}

print("\n=== Portfolio Greeks Summary ===")
for name, (d, g, v, _) in portfolios.items():
    print(f"{name:25s}: Delta={d:12.6f}, Gamma={g:12.6e}, Vega={v:12.6f}")
    if abs(d) < GREEK_TOLERANCE:
        print(f"  WARNING: {name} has near-zero delta - VaR may be very small!")
    if abs(g) < GREEK_TOLERANCE:
        print(f"  WARNING: {name} has near-zero gamma - quadratic effects negligible!")

# ---------- Calculate VaR for all portfolios ----------
parametric_rows = []
hist_rows = []
all_pnl_series = {}

for port_name, (port_delta, port_gamma, port_vega, port_positions) in portfolios.items():
    
    # Check if portfolio is effectively risk-free
    if abs(port_delta) < GREEK_TOLERANCE and abs(port_gamma) < GREEK_TOLERANCE:
        print(f"\n⚠️  {port_name} has near-zero delta and gamma!")
        print(f"   This portfolio is effectively hedged against small price moves.")
        print(f"   Remaining risk: Vega={port_vega:.6f}, Higher-order effects, Model risk")
    
    # Parametric VaR
    if abs(port_delta) > GREEK_TOLERANCE or abs(port_gamma) < GREEK_TOLERANCE:
        # Linear portfolio - use delta-normal
        std_pnl_simple = abs(port_delta * S_spot * sigma_simple)
        std_pnl_log = abs(port_delta * S_spot * sigma_log)
        
        if INCLUDE_VEGA_IN_VAR and abs(port_vega) > GREEK_TOLERANCE:
            vol_std = np.std(vol_changes, ddof=1)
            std_pnl_simple = np.sqrt(std_pnl_simple**2 + (port_vega * vol_std / 100)**2)
            std_pnl_log = np.sqrt(std_pnl_log**2 + (port_vega * vol_std / 100)**2)
        
        parametric_rows.append({
            'Portfolio': port_name, 'ReturnType': 'simple', 'Method': 'Parametric',
            'VaR95': std_pnl_simple * z95, 'VaR99': std_pnl_simple * z99
        })
        parametric_rows.append({
            'Portfolio': port_name, 'ReturnType': 'log', 'Method': 'Parametric',
            'VaR95': std_pnl_log * z95, 'VaR99': std_pnl_log * z99
        })
    else:
        # Nonlinear portfolio - use MC
        np.random.seed(42 + len(parametric_rows))
        mc_returns_simple = np.random.normal(0, sigma_simple, MC_DRAWS)
        mc_vol_changes = np.random.normal(0, np.std(vol_changes, ddof=1), MC_DRAWS) if INCLUDE_VEGA_IN_VAR else np.zeros(MC_DRAWS)
        pnl_mc_simple = pnl_quadratic(port_delta, port_gamma, port_vega, mc_returns_simple, mc_vol_changes, S_spot)
        
        mc_returns_log = np.random.normal(0, sigma_log, MC_DRAWS)
        pnl_mc_log = pnl_quadratic(port_delta, port_gamma, port_vega, mc_returns_log, mc_vol_changes, S_spot)
        
        parametric_rows.append({
            'Portfolio': port_name, 'ReturnType': 'simple', 'Method': 'Parametric_MC',
            'VaR95': -np.percentile(pnl_mc_simple, 5), 'VaR99': -np.percentile(pnl_mc_simple, 1)
        })
        parametric_rows.append({
            'Portfolio': port_name, 'ReturnType': 'log', 'Method': 'Parametric_MC',
            'VaR95': -np.percentile(pnl_mc_log, 5), 'VaR99': -np.percentile(pnl_mc_log, 1)
        })
    
    # Historical VaR
    if USE_FULL_REPRICING:
        v95_s, v99_s, pnl_s = historical_var_full_repricing(port_positions, greeks_df, S_spot, returns_simple, vol_changes)
        v95_l, v99_l, pnl_l = historical_var_full_repricing(port_positions, greeks_df, S_spot, returns_log, vol_changes)
    else:
        pnl_s = pnl_quadratic(port_delta, port_gamma, port_vega, returns_simple, vol_changes, S_spot)
        pnl_l = pnl_quadratic(port_delta, port_gamma, port_vega, returns_log, vol_changes, S_spot)
        v95_s, v99_s = -np.percentile(pnl_s, 5), -np.percentile(pnl_s, 1)
        v95_l, v99_l = -np.percentile(pnl_l, 5), -np.percentile(pnl_l, 1)
    
    hist_rows.append({
        'Portfolio': port_name, 'ReturnType': 'simple', 
        'Method': 'Historical_FullRepricing' if USE_FULL_REPRICING else 'Historical',
        'VaR95': v95_s, 'VaR99': v99_s
    })
    hist_rows.append({
        'Portfolio': port_name, 'ReturnType': 'log',
        'Method': 'Historical_FullRepricing' if USE_FULL_REPRICING else 'Historical',
        'VaR95': v95_l, 'VaR99': v99_l
    })
    
    all_pnl_series[f'{port_name}_simple'] = pnl_s
    all_pnl_series[f'{port_name}_log'] = pnl_l

# ---------- Save outputs ----------
os.makedirs(OUT_DIR, exist_ok=True)
parametric_df = pd.DataFrame(parametric_rows)
hist_df = pd.DataFrame(hist_rows)

parametric_df.to_csv(os.path.join(OUT_DIR, "parametric_var_FIXED.csv"), index=False)
hist_df.to_csv(os.path.join(OUT_DIR, "historical_var_FIXED.csv"), index=False)
pd.DataFrame(all_pnl_series).to_csv(os.path.join(OUT_DIR, "pnl_series_FIXED.csv"), index=False)

print("\n=== VaR Results Summary (95% confidence) ===")
print("\nParametric VaR:")
print(parametric_df[parametric_df['ReturnType']=='simple'][['Portfolio','Method','VaR95']].to_string(index=False))
print("\nHistorical VaR:")
print(hist_df[hist_df['ReturnType']=='simple'][['Portfolio','Method','VaR95']].to_string(index=False))

print("\n✅ Fixed VaR calculations saved!")
print("\n💡 Key insights:")
print("1. Delta+Gamma hedged portfolio SHOULD have low VaR (that's the point!)")
print("2. But it's not zero due to: vega risk, higher-order Greeks, discrete hedging")
print("3. Full repricing captures effects that Greek approximation misses")
