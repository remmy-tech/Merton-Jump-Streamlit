import streamlit as st
import numpy as np
import pandas as pd
import math
import datetime as dt
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf

# ---------------------- Pricing Functions ---------------------- #

# Black-Scholes functions
def call_BS(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_BS(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# Merton Jump Diffusion functions (Corrected)
def merton_jump_call(S, K, T, r, sigma, m, v, lam, N_terms=40):
    price = 0
    poisson_weight = np.exp(-lam * T)
    for k in range(N_terms):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
        weight = poisson_weight * (lam * T) ** k / math.factorial(k)
        price += weight * call_BS(S, K, T, r_k, sigma_k)
    return price

def merton_jump_put(S, K, T, r, sigma, m, v, lam, N_terms=40):
    price = 0
    poisson_weight = np.exp(-lam * T)
    for k in range(N_terms):
        r_k = r - lam * (m - 1) + (k * np.log(m)) / T
        sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
        weight = poisson_weight * (lam * T) ** k / math.factorial(k)
        price += weight * put_BS(S, K, T, r_k, sigma_k)
    return price

# ---------------------- Streamlit App Layout ---------------------- #

st.title("Option Pricing Models")

# Sidebar: Choose Model
model_choice = st.sidebar.radio("Select Pricing Model", ["Black–Scholes", "Merton Jump Diffusion"])

# Common parameters
st.sidebar.header("Common Parameters")
S = st.sidebar.number_input("Stock Price (S)", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.02, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)

# Additional parameters for Merton Jump Diffusion
if model_choice == "Merton Jump Diffusion":
    st.sidebar.header("Jump Parameters")
    m = st.sidebar.number_input("Mean Jump Multiplier (m)", min_value=0.01, value=1.0, step=0.01)
    v = st.sidebar.number_input("Jump Volatility (v)", min_value=0.01, value=0.3, step=0.01)
    lam = st.sidebar.number_input("Jump Intensity (λ)", min_value=0.01, value=1.0, step=0.01)
    N_terms = st.sidebar.number_input("Number of Poisson Terms", min_value=10, max_value=100, value=40, step=1)
else:
    # Set dummy values for Black–Scholes
    m, v, lam, N_terms = 1.0, 0.0, 0.0, 0

# ---------------------- Compute Option Prices ---------------------- #
if model_choice == "Black–Scholes":
    call_price = call_BS(S, K, T, r, sigma)
    put_price = put_BS(S, K, T, r, sigma)
else:
    call_price = merton_jump_call(S, K, T, r, sigma, m, v, lam, N_terms)
    put_price = merton_jump_put(S, K, T, r, sigma, m, v, lam, N_terms)

# Display Option Prices in a Two-Column Layout
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"{model_choice} Call Price", value=f"${call_price:.2f}")
with col2:
    st.metric(label=f"{model_choice} Put Price", value=f"${put_price:.2f}")

# ---------------------- Interactive Plot: Option Price vs. Strike ---------------------- #
st.header("Option Price Across Different Strike Prices")

# Define a range for strikes
strike_range = np.arange(K * 0.5, K * 1.5, 1)
if model_choice == "Black–Scholes":
    call_prices = [call_BS(S, strike, T, r, sigma) for strike in strike_range]
    put_prices = [put_BS(S, strike, T, r, sigma) for strike in strike_range]
else:
    call_prices = [merton_jump_call(S, strike, T, r, sigma, m, v, lam, N_terms) for strike in strike_range]
    put_prices = [merton_jump_put(S, strike, T, r, sigma, m, v, lam, N_terms) for strike in strike_range]

fig = go.Figure()
fig.add_trace(go.Scatter(x=strike_range, y=call_prices, mode='lines', name='Call Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=strike_range, y=put_prices, mode='lines', name='Put Price', line=dict(color='red')))
fig.add_vline(x=S, line=dict(color='black', dash='dash'), annotation_text="Stock Price", annotation_position="top left")
fig.update_layout(title=f"{model_choice} Option Prices vs. Strike",
                  xaxis_title="Strike Price (K)",
                  yaxis_title="Option Price")
st.plotly_chart(fig, use_container_width=True)

# ---------------------- Expanders for Additional Details ---------------------- #
with st.expander("View Code Details / Methodology"):
    st.markdown("""
    **Model Details:**
    - **Black–Scholes:** Uses the standard closed-form solution for European options.
    - **Merton Jump Diffusion:** Adjusts the drift and volatility in each Poisson-weighted term to account for jumps.
    
    **Note:** In the Merton model, the number of jumps follows a Poisson process with mean \\( \\lambda T \\), and the drift is adjusted by \\( -\\lambda(m-1) \\) while each term’s volatility is increased by \\( \\sqrt{\\sigma^2 + \\frac{k\\,v^2}{T}} \\).
    """)