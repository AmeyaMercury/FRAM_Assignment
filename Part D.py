"""
Option Portfolio Construction, Hedging, and PnL Simulation
Assumes an options CSV with columns:
  Strike, Strike_Price, Maturity_Days, Call_Price, Put_Price

Author: ChatGPT (code)
Usage: edit `CSV_PATH` and parameters in the example at the bottom, then run.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# --------------------------
# Black-Scholes pricing + greeks
# --------------------------
def bs_price_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_price_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol_call(market_price, S, K, T, r, bracket=(1e-6, 5.0)):
    """Invert BS to get implied vol for call using Brent root-finder.
       Returns np.nan on failure.
    """
    if market_price <= max(S - K, 0.0) - 1e-12:
        # Market price below intrinsic (bad data) — return nan
        return np.nan
    try:
        f = lambda sigma: bs_price_call(S, K, T, r, sigma) - market_price
        return brentq(f, bracket[0], bracket[1], maxiter=200, xtol=1e-6)
    except Exception:
        return np.nan

def greeks_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or math.isnan(sigma):
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def greeks_put_from_call(S, K, T, r, sigma):
    g = greeks_call(S, K, T, r, sigma)
    # put-call parity relations for Greeks:
    # Gamma, Vega equal; Delta_put = Delta_call - 1
    delta_put = g['delta'] - 1.0
    # Theta and Rho for put can be derived, but we only need delta/gamma/vega primarily
    theta_put = g['theta'] + r * K * math.exp(-r * T)
    rho_put = g['rho'] - K * T * math.exp(-r * T)
    return dict(delta=delta_put, gamma=g['gamma'], vega=g['vega'], theta=theta_put, rho=rho_put)

# --------------------------
# Utility: load & prepare options data
# --------------------------
def load_options(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize column names (small conveniences)
    df.columns = [c.strip() for c in df.columns]
    required = {'Strike', 'Strike_Price', 'Maturity_Days', 'Call_Price', 'Put_Price'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Found: {df.columns}")
    return df

# --------------------------
# Build instruments data (IV + greeks)
# --------------------------
def compute_ivs_and_greeks(df, S, r):
    rows = []
    for _, row in df.iterrows():
        K = float(row['Strike_Price'])
        T = float(row['Maturity_Days']) / 365.0
        call_mkt = float(row['Call_Price'])
        put_mkt = float(row['Put_Price'])
        iv = implied_vol_call(call_mkt, S, K, T, r)
        cgreeks = greeks_call(S, K, T, r, iv) if not math.isnan(iv) else dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
        pgreeks = greeks_put_from_call(S, K, T, r, iv) if not math.isnan(iv) else dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
        rows.append({
            'StrikeLabel': row['Strike'],
            'K': K,
            'T_days': int(row['Maturity_Days']),
            'Call_Price': call_mkt,
            'Put_Price': put_mkt,
            'ImpliedVol': iv,
            'Call_delta': cgreeks['delta'],
            'Call_gamma': cgreeks['gamma'],
            'Call_vega': cgreeks['vega'],
            'Put_delta': pgreeks['delta'],
            'Put_gamma': pgreeks['gamma'],
            'Put_vega': pgreeks['vega']
        })
    return pd.DataFrame(rows)

# --------------------------
# Portfolio functions
# --------------------------
def build_portfolio_from_spec(instruments_df, spec):
    """
    spec: dict mapping instrument identifiers to quantity.
    instrument identifier format examples:
      'Call_ATM_30' or 'Put_ATM_60' or any string you choose that matches entries we create below.
    Returns a DataFrame with portfolio line items.
    """
    # Create instrument name map from instruments_df
    instr_map = {}
    for _, r in instruments_df.iterrows():
        # define names for call and put per row
        name_call = f"Call_{r['StrikeLabel']}_{int(r['T_days'])}d"
        name_put  = f"Put_{r['StrikeLabel']}_{int(r['T_days'])}d"
        instr_map[name_call] = dict(Type='Call', K=r['K'], T_days=r['T_days'], Price=r['Call_Price'],
                                    Delta=r['Call_delta'], Gamma=r['Call_gamma'], Vega=r['Call_vega'])
        instr_map[name_put]  = dict(Type='Put',  K=r['K'], T_days=r['T_days'], Price=r['Put_Price'],
                                    Delta=r['Put_delta'], Gamma=r['Put_gamma'], Vega=r['Put_vega'])
    # Build portfolio DataFrame
    port_rows = []
    for instr_name, qty in spec.items():
        if instr_name not in instr_map:
            raise ValueError(f"Instrument '{instr_name}' not found in instrument map.")
        info = instr_map[instr_name]
        port_rows.append({
            'Instrument': instr_name,
            'Type': info['Type'],
            'K': info['K'],
            'T_days': info['T_days'],
            'Qty': qty,
            'Price': info['Price'],
            'Delta': info['Delta'],
            'Gamma': info['Gamma'],
            'Vega': info['Vega']
        })
    port_df = pd.DataFrame(port_rows)
    # contributions
    port_df['Delta_contrib'] = port_df['Qty'] * port_df['Delta']
    port_df['Gamma_contrib'] = port_df['Qty'] * port_df['Gamma']
    port_df['Vega_contrib']  = port_df['Qty'] * port_df['Vega']
    return port_df

def portfolio_summary(port_df):
    return {
        'Total_Value': (port_df['Qty'] * port_df['Price']).sum(),
        'Total_Delta': port_df['Delta_contrib'].sum(),
        'Total_Gamma': port_df['Gamma_contrib'].sum(),
        'Total_Vega':  port_df['Vega_contrib'].sum()
    }

# --------------------------
# Hedging (Delta + Gamma) helpers
# --------------------------
def compute_delta_hedge_stock_qty(total_delta):
    # if we want delta-neutral w.r.t underlying: hold -total_delta shares
    return -total_delta

def compute_gamma_hedge_calls_qty(total_gamma, hedge_call_gamma):
    # number of calls to buy/sell to neutralize gamma
    if hedge_call_gamma == 0 or np.isnan(hedge_call_gamma):
        return 0.0
    return - total_gamma / hedge_call_gamma

# --------------------------
# PnL simulation under shocks
# --------------------------
def repriced_option_price(S_new, instr_type, K, T_days, r, iv):
    T = float(T_days) / 365.0
    if instr_type == 'Call':
        return bs_price_call(S_new, K, T, r, iv)
    else:
        return bs_price_put(S_new, K, T, r, iv)

def simulate_pnl_under_shocks(port_df, instruments_df, S, r, shocks=[0.01, -0.01, 0.02, -0.02]):
    """
    Uses constant implied vol (from instruments_df) and reprices options at S * (1+shock).
    Returns a DataFrame with PnL for each shock for:
      - Unhedged portfolio
      - Delta hedged (stock)
      - Gamma+Delta hedged (calls + stock)
    """
    # create map for implied vol and base price in instruments_df
    # Map keys consistent with build_portfolio_from_spec naming
    iv_map = {}
    for _, r0 in instruments_df.iterrows():
        name_call = f"Call_{r0['StrikeLabel']}_{int(r0['T_days'])}d"
        name_put  = f"Put_{r0['StrikeLabel']}_{int(r0['T_days'])}d"
        iv_map[name_call] = float(r0['ImpliedVol']) if not np.isnan(r0['ImpliedVol']) else None
        iv_map[name_put]  = float(r0['ImpliedVol']) if not np.isnan(r0['ImpliedVol']) else None

    # initial portfolio value
    initial_value = (port_df['Qty'] * port_df['Price']).sum()

    # compute portfolio Greeks totals
    total_delta = port_df['Delta_contrib'].sum()
    total_gamma = port_df['Gamma_contrib'].sum()
    total_vega  = port_df['Vega_contrib'].sum()

    # Choose gamma hedge instrument (default: 30-day ATM Call if present)
    # find a call with T_days == 30 and StrikeLabel containing 'ATM'
    hedge_call_row = None
    for _, r0 in instruments_df.iterrows():
        if (r0['StrikeLabel'].upper().find('ATM') >= 0) and int(r0['T_days']) == 30:
            hedge_call_row = r0
            break
    if hedge_call_row is None:
        # fallback: pick the first call in instruments_df
        hedge_call_row = instruments_df.iloc[0]

    hedge_call_name = f"Call_{hedge_call_row['StrikeLabel']}_{int(hedge_call_row['T_days'])}d"
    hedge_call_gamma = float(hedge_call_row['Call_gamma'])
    hedge_call_delta = float(hedge_call_row['Call_delta'])
    hedge_call_iv = float(hedge_call_row['ImpliedVol'])
    hedge_call_price = float(hedge_call_row['Call_Price'])

    # compute hedge quantities
    n_gamma_calls = compute_gamma_hedge_calls_qty(total_gamma, hedge_call_gamma)
    # delta after buying gamma hedge calls:
    delta_after_gamma_calls = total_delta + n_gamma_calls * hedge_call_delta
    # stock needed to neutralize delta after gamma hedge:
    stock_shares_after = -delta_after_gamma_calls
    # delta-only hedge (stock to neutralize initial total_delta)
    stock_shares_for_delta_hedge = compute_delta_hedge_stock_qty(total_delta)

    results = []
    for shock in shocks:
        S_new = S * (1.0 + shock)
        # Reprice each instrument at S_new using constant IV
        pv_new = 0.0
        for _, row in port_df.iterrows():
            instr_name = row['Instrument']
            qty = row['Qty']
            instr_type = row['Type']
            K = row['K']; T_days = row['T_days']
            iv = iv_map[instr_name]
            if iv is None or np.isnan(iv):
                # if IV missing, fall back to last price (assume little/no change) or skip
                new_price = row['Price']
            else:
                new_price = repriced_option_price(S_new, instr_type, K, T_days, r, iv)
            pv_new += qty * new_price

        pnl_unhedged = pv_new - initial_value

        # delta hedge PnL: include stock position opened at S
        pnl_delta_hedged = pnl_unhedged + stock_shares_for_delta_hedge * (S_new - S)

        # gamma+delta hedge PnL: include gamma call position and final stock position
        # value change of hedge calls
        if n_gamma_calls != 0.0 and (hedge_call_iv is not None and not np.isnan(hedge_call_iv)):
            hedge_call_new_price = bs_price_call(S_new, hedge_call_row['K'], hedge_call_row['T_days']/365.0, r, hedge_call_iv)
            hedge_calls_pnl = n_gamma_calls * (hedge_call_new_price - hedge_call_price)
        else:
            hedge_calls_pnl = 0.0
        pnl_gamma_delta_hedged = pnl_unhedged + hedge_calls_pnl + stock_shares_after * (S_new - S)

        results.append({
            'Shock': shock,
            'S_new': S_new,
            'Unhedged_PnL': pnl_unhedged,
            'Delta_Hedged_PnL': pnl_delta_hedged,
            'Gamma+Delta_Hedged_PnL': pnl_gamma_delta_hedged,
            'GammaCallsQty': n_gamma_calls,
            'StockShares_for_Delta_Hedge': stock_shares_for_delta_hedge,
            'StockShares_after_Gamma_then_Delta': stock_shares_after
        })

    return pd.DataFrame(results)

# --------------------------
# Example usage (bottom of file) - edit before running
# --------------------------
if __name__ == "__main__":
    # USER SETTINGS - edit:
    CSV_PATH = "TITAN_BSM_Options.csv"   # path to your options csv
    RISK_FREE_RATE = 0.06513            # e.g., 6.513% -> 0.06513
    # Portfolio specification: dictionary mapping instrument-name -> quantity
    # instrument-name examples: "Call_ATM_30d", "Put_ATM_60d" etc.
    # We'll auto-generate those names from the CSV; list them below once computed.
    # Example portfolio: long 1 ATM call & short 1 ATM put for each maturity (30,60,90)
    EXAMPLE_PORT_SPEC = {
        # Use the exact labels produced by compute_ivs_and_greeks + build_portfolio_from_spec
        # We'll auto-create an example if you prefer consistent naming:
    }

    # ---- load options
    raw_opts = load_options(CSV_PATH)

    # detect S (spot) from ATM row (Strike == 'ATM') else use last close or provided spot
    atm_rows = raw_opts[raw_opts['Strike'].astype(str).str.upper().str.contains('ATM')]
    if atm_rows.shape[0] > 0:
        spot = float(atm_rows.iloc[0]['Strike_Price'])
    else:
        # fallback: use middle strike or let user set spot manually
        spot = float(raw_opts['Strike_Price'].iloc[0])

    # compute IVs + greeks table
    instr_df = compute_ivs_and_greeks(raw_opts, spot, RISK_FREE_RATE)
    print("Instruments computed (IVs & Greeks):")
    print(instr_df[['StrikeLabel','K','T_days','ImpliedVol','Call_delta','Call_gamma','Call_vega','Put_delta','Put_gamma','Put_vega']])

    # show the instrument names you can use in portfolio spec
    avail_names = []
    for _, r in instr_df.iterrows():
        avail_names.append(f"Call_{r['StrikeLabel']}_{int(r['T_days'])}d")
        avail_names.append(f"Put_{r['StrikeLabel']}_{int(r['T_days'])}d")
    print("\nAvailable instrument names to use in spec:")
    for name in avail_names:
        print("  ", name)

    # If user left EXAMPLE_PORT_SPEC empty, create the default example: +1 call, -1 put for ATM for each maturity
    if not EXAMPLE_PORT_SPEC:
        EXAMPLE_PORT_SPEC = {}
        for _, r in instr_df[instr_df['StrikeLabel'].astype(str).str.upper().str.contains('ATM')].iterrows():
            EXAMPLE_PORT_SPEC[f"Call_{r['StrikeLabel']}_{int(r['T_days'])}d"] = EXAMPLE_PORT_SPEC.get(f"Call_{r['StrikeLabel']}_{int(r['T_days'])}d", 0) + 1
            EXAMPLE_PORT_SPEC[f"Put_{r['StrikeLabel']}_{int(r['T_days'])}d"]  = EXAMPLE_PORT_SPEC.get(f"Put_{r['StrikeLabel']}_{int(r['T_days'])}d", 0) - 1

    print("\nPortfolio spec (instrument -> qty):")
    print(EXAMPLE_PORT_SPEC)

    # Build portfolio DataFrame
    port_df = build_portfolio_from_spec(instr_df, EXAMPLE_PORT_SPEC)
    print("\nPortfolio lines:")
    print(port_df[['Instrument','Qty','Price','Delta','Gamma','Vega','Delta_contrib','Gamma_contrib','Vega_contrib']])

    # Portfolio summary
    summary = portfolio_summary(port_df)
    print("\nPortfolio summary (before hedging):")
    print(summary)

    # Compute hedges
    stock_delta_qty = compute_delta_hedge_stock_qty(summary['Total_Delta'])
    # gamma hedge using 30d ATM call (selected in simulation function)
    # find 30d ATM call inside instr_df
    hedge_candidate = instr_df[(instr_df['StrikeLabel'].astype(str).str.upper().str.contains('ATM')) & (instr_df['T_days']==30)]
    if hedge_candidate.shape[0] > 0:
        hedge_call_gamma = float(hedge_candidate.iloc[0]['Call_gamma'])
        hedge_call_delta = float(hedge_candidate.iloc[0]['Call_delta'])
        hedge_call_price = float(hedge_candidate.iloc[0]['Call_Price'])
        hedge_call_name = f"Call_{hedge_candidate.iloc[0]['StrikeLabel']}_30d"
    else:
        # fallback to first call instrument
        hedge_call_gamma = float(instr_df.iloc[0]['Call_gamma'])
        hedge_call_delta = float(instr_df.iloc[0]['Call_delta'])
        hedge_call_price = float(instr_df.iloc[0]['Call_Price'])
        hedge_call_name = f"Call_{instr_df.iloc[0]['StrikeLabel']}_{int(instr_df.iloc[0]['T_days'])}d"

    n_gamma_calls = compute_gamma_hedge_calls_qty(summary['Total_Gamma'], hedge_call_gamma)
    delta_after_adding_gamma = summary['Total_Delta'] + n_gamma_calls * hedge_call_delta
    stock_after_full_hedge = -delta_after_adding_gamma

    print("\nHedge plan:")
    print(f"  Stock shares to delta-hedge initially: {stock_delta_qty:.6f}")
    print(f"  Gamma hedge: buy {n_gamma_calls:.6f} x {hedge_call_name} (each priced {hedge_call_price})")
    print(f"  Delta after gamma calls: {delta_after_adding_gamma:.6f}")
    print(f"  Stock shares after full (gamma+delta) hedge: {stock_after_full_hedge:.6f}")

    # Show Greeks before and after hedge
    before = dict(Delta=summary['Total_Delta'], Gamma=summary['Total_Gamma'], Vega=summary['Total_Vega'])
    after_gamma = dict(Delta=summary['Total_Delta'] + n_gamma_calls * hedge_call_delta,
                       Gamma=summary['Total_Gamma'] + n_gamma_calls * hedge_call_gamma,
                       Vega=summary['Total_Vega'] + n_gamma_calls * float(hedge_candidate.iloc[0]['Call_vega']) if hedge_candidate.shape[0]>0 else summary['Total_Vega'])
    after_full = dict(Delta=after_gamma['Delta'] + stock_after_full_hedge, Gamma=after_gamma['Gamma'], Vega=after_gamma['Vega'])

    print("\nGreeks (before hedge):", before)
    print("Greeks (after gamma hedge):", {k: round(v,6) for k,v in after_gamma.items()})
    print("Greeks (after gamma+delta hedge):", {k: round(v,6) for k,v in after_full.items()})

    # Simulate PnL under shocks
    shocks = [0.01, -0.01, 0.02, -0.02]
    pnl_df = simulate_pnl_under_shocks(port_df, instr_df, spot, RISK_FREE_RATE, shocks=shocks)
    print("\nPnL simulation under shocks (unhedged, delta-hedged, gamma+delta-hedged):")
    print(pnl_df[['Shock','S_new','Unhedged_PnL','Delta_Hedged_PnL','Gamma+Delta_Hedged_PnL']])

    # Save outputs for your report
    port_df.to_csv("portfolio_positions.csv", index=False)
    pd.DataFrame([before, after_gamma, after_full], index=['before','after_gamma','after_full']).to_csv("greeks_summary.csv")
    pnl_df.to_csv("pnl_simulation.csv", index=False)
    print("\nSaved outputs: portfolio_positions.csv, greeks_summary.csv, pnl_simulation.csv")
