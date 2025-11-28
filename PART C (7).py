import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp
from tvDatafeed import TvDatafeed, Interval

# ---------------------------
# USER INPUTS
# ---------------------------
tv = TvDatafeed()  # Initialize TvDatafeed
S0 =  float(tv.get_hist('TITAN', 'NSE').close.iloc[-1])    # Underlying price
r = 0.07         # Risk-free annual rate
# ---------------------------

# Black–Scholes pricing
def bs_call(S, K, T, r, sigma):
    if T <= 0: 
        return max(S-K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K-S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


# ---------------------------
# IMPLIED VOLATILITY SOLVER
# ---------------------------
def implied_vol_call(C_obs, S, K, T, r, tol=1e-6, max_iter=100):
    sigma = 0.20  # initial guess
    for i in range(max_iter):
        price = bs_call(S, K, T, r, sigma)
        vega = S * norm.pdf((log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))) * sqrt(T)
        diff = price - C_obs
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega if vega > 1e-8 else 0.0001
        sigma = max(sigma, 1e-6)
    return sigma


def implied_vol_put(P_obs, S, K, T, r, tol=1e-6, max_iter=100):
    sigma = 0.20
    for i in range(max_iter):
        price = bs_put(S, K, T, r, sigma)
        vega = S * norm.pdf((log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))) * sqrt(T)
        diff = price - P_obs
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega if vega > 1e-8 else 0.0001
        sigma = max(sigma, 1e-6)
    return sigma


# ---------------------------
# GREEK FUNCTIONS
# ---------------------------
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S,K,T,r,sigma))

def put_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S,K,T,r,sigma)) - 1

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S,K,T,r,sigma)) / (S*sigma*np.sqrt(T))

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S,K,T,r,sigma)) * np.sqrt(T) / 100

def call_theta(S, K, T, r, sigma):
    term1 = -(S * norm.pdf(d1(S,K,T,r,sigma)) * sigma) / (2*np.sqrt(T))
    term2 = -r*K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
    return (term1 + term2) / 365

def put_theta(S, K, T, r, sigma):
    term1 = -(S * norm.pdf(d1(S,K,T,r,sigma)) * sigma) / (2*np.sqrt(T))
    term2 = +r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma))
    return (term1 + term2) / 365

def call_rho(S, K, T, r, sigma):
    return (K*T*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))) / 100

def put_rho(S, K, T, r, sigma):
    return (-K*T*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma))) / 100


# ---------------------------
# LOAD CSV AND CALCULATE EVERYTHING
# ---------------------------
df = pd.read_csv('TITAN_BSM_Options.csv')
df['T'] = df['Maturity_Days'] / 365

# Compute IV per row (Call IV is used)
df['IV'] = df.apply(lambda x: implied_vol_call(
    x['Call_Price'], S0, x['Strike_Price'], x['T'], r
), axis=1)

# Greeks using per-row IV
df['Call_Delta'] = df.apply(lambda x: call_delta(S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Put_Delta']  = df.apply(lambda x: put_delta (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Gamma']      = df.apply(lambda x: gamma     (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Vega']       = df.apply(lambda x: vega      (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Call_Theta'] = df.apply(lambda x: call_theta(S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Put_Theta']  = df.apply(lambda x: put_theta (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Call_Rho']   = df.apply(lambda x: call_rho  (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)
df['Put_Rho']    = df.apply(lambda x: put_rho   (S0, x['Strike_Price'], x['T'], r, x['IV']), axis=1)

# Save output
output_path = 'titan_greeks.csv'
df.to_csv(output_path, index=False)

output_path
