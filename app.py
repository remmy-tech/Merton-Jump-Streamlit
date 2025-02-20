import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import datetime as dt
import plotly.graph_objects as go
import yfinance as yf
import math

# Normal CDF function
N = norm.cdf
# ---------------------- Black and Sholes Functions ---------------------- #

def call_BS(S, K, T, r, sigma):
    """ Black-Scholes Call Option Pricing Formula """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)

def put_BS(S, K, T, r, sigma):
    """ Black-Scholes Put Option Pricing Formula """
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * N(-d2) - S * N(-d1)
# ---------------------- Merton Jump Diffusion Functions ---------------------- #

def merton_jump_call(S, K, T, r, sigma, m, v, lam, N_terms=40):
    """ Merton Jump Diffusion Model for European Call option pricing. """
    price = 0
    exp_neg_mlamT = np.exp(-m * lam * T)  

    for k in range(N_terms):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma**2 + (k * v**2) / T)
        weight = exp_neg_mlamT * (m * lam * T)**k / math.factorial(k)
        price += weight * call_BS(S, K, T, r_k, sigma_k)
    
    return price

def merton_jump_put(S, K, T, r, sigma, m, v, lam, N_terms=40):
    """ Merton Jump Diffusion Model for European Put option pricing. """
    price = 0
    exp_neg_mlamT = np.exp(-m * lam * T)

    for k in range(N_terms):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma**2 + (k * v**2) / T)
        weight = exp_neg_mlamT * (m * lam * T)**k / math.factorial(k)
        price += weight * put_BS(S, K, T, r_k, sigma_k)
    
    return price

# ---------------------- Greeks Calculation ---------------------- #

def delta_call(S, K, T, r, sigma, m, v, lam):
    """ Delta for Call under Merton Jump Diffusion """
    return (merton_jump_call(S + 1, K, T, r, sigma, m, v, lam) - merton_jump_call(S, K, T, r, sigma, m, v, lam))

def delta_put(S, K, T, r, sigma, m, v, lam):
    """ Delta for Put under Merton Jump Diffusion """
    return (merton_jump_put(S + 1, K, T, r, sigma, m, v, lam) - merton_jump_put(S, K, T, r, sigma, m, v, lam))

# ---------------------- Streamlit App ---------------------- #

# Title
st.title("Merton Jump Diffusion Option Pricing Model")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input('Stock Price (S)', min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input('Strike Price (K)', min_value=1.0, value=100.0, step=1.0)
exp = st.sidebar.date_input('Expiry Date', value=dt.datetime(2025, 9, 19))
exp = dt.datetime.combine(exp, dt.datetime.min.time())
T = (exp - dt.datetime.today()).days / 365
r = st.sidebar.number_input('Risk-Free Rate (r)', min_value=0.0, value=0.02, step=0.01)
sigma = st.sidebar.number_input('Volatility (σ)', min_value=0.01, value=0.2, step=0.01)
m = st.sidebar.number_input('Mean Jump Size (m)', min_value=0.01, value=1.0, step=0.01)
v = st.sidebar.number_input('Jump Size Std Dev (v)', min_value=0.01, value=0.3, step=0.01)
lam = st.sidebar.number_input('Jump Intensity (λ)', min_value=0.01, value=1.0, step=0.01)

# Compute Option Prices
call_price_JDM = merton_jump_call(S, K, T, r, sigma, m, v, lam)
put_price_JDM = merton_jump_put(S, K, T, r, sigma, m, v, lam)

# Display Prices
col1, col2 = st.columns(2)
with col1: st.metric(label='Merton Call Option Price', value=f"${call_price_JDM:.2f}")
with col2: st.metric(label='Merton Put Option Price', value=f"${put_price_JDM:.2f}")

# Plot Option Prices Across Different Strikes
strikes = np.arange(50, 150, 1)
call_prices = [merton_jump_call(S, K, T, r, sigma, m, v, lam) for K in strikes]
put_prices = [merton_jump_put(S, K, T, r, sigma, m, v, lam) for K in strikes]

plt.figure(figsize=(10, 5))
plt.plot(strikes, call_prices, label='MJD Call Price', color='blue')
plt.plot(strikes, put_prices, label='MJD Put Price', color='red')
plt.axvline(S, color='black', linestyle='dashed', label='Stock Price (S)')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')
plt.title('Merton Jump Diffusion Option Prices')
plt.legend()
st.pyplot(plt)

# ---------------------- Historical Volatility Comparison ---------------------- #
st.header("Compare with Historical Volatility")
ticker = st.text_input('Yahoo Stock Ticker', value='AAPL')
start = st.date_input('Start Date', value=dt.datetime(2021, 1, 1))
end = st.date_input('End Date', value=dt.datetime.today())

# Fetch Stock Data
stockData = yf.download(ticker, start, end)
stockData['dReturns'] = stockData['Close'].pct_change()
stockData['HVol'] = stockData['dReturns'].rolling(window=30).std() * np.sqrt(252)

# Plot Historical vs Implied Volatility
fig = go.Figure()
fig.add_trace(go.Scatter(x=stockData.index, y=stockData['HVol'], mode='lines', name='Historical Volatility', line=dict(color='lightblue')))
fig.add_trace(go.Scatter(x=[stockData.index.min(), stockData.index.max()], y=[sigma, sigma], mode='lines', name='Implied Volatility', line=dict(color='red')))

fig.update_layout(title='Historical and Implied Volatility', xaxis_title='Date', yaxis_title='Volatility')
st.plotly_chart(fig, use_container_width=True)