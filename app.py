import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import datetime as dt
from scipy.stats import norm

# ------------- Black-Scholes functions (used inside MJD expansion) ------------- #
def call_BS(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes price of a European call.
    """
    if T <= 0:
        return max(S - K, 0)  # Expired
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def put_BS(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes price of a European put.
    """
    if T <= 0:
        return max(K - S, 0)  # Expired
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)


# ------------- Merton Jump-Diffusion (MJD) pricing functions ------------- #
def merton_jump_call(S, K, T, r, sigma, q, m, v, lam, N=40):
    """
    European call under Merton's Jump-Diffusion using summation of
    Poisson-weighted Black–Scholes prices.
      - lam = jump intensity (λ)
      - m   = e^(mean jump), or "mean jump size" > 0
      - v   = jump volatility (std dev of jump)
    We assume risk-neutral drift shift: r -> (r - q - λ*(m - 1)).
    """
    if T <= 0:
        return max(S - K, 0)
    price = 0.0
    # Poisson pmf factor: e^{-lam*T} (lam*T)^k / k!
    for k in range(N):
        w_k = np.exp(-lam*T)*( (lam*T)**k / math.factorial(k) )
        # Adjust underlying upward by (k jumps), each jump ~ ln(m)
        S_k = S * np.exp(k * np.log(m))
        # Adjust volatility: sqrt( sigma^2 + k*v^2 )
        sigma_k = np.sqrt( sigma**2 + k*(v**2) )
        # Effective drift: (r - q - lam*(m - 1))
        r_k = (r - q) - lam*(m - 1)
        # Price of call for that scenario
        bs_price = call_BS(S_k, K, T, r_k, sigma_k, q=0.0)
        price += w_k * bs_price
    return price

def merton_jump_put(S, K, T, r, sigma, q, m, v, lam, N=40):
    """
    European put under Merton's Jump-Diffusion, same logic as above.
    """
    if T <= 0:
        return max(K - S, 0)
    price = 0.0
    for k in range(N):
        w_k = np.exp(-lam*T)*( (lam*T)**k / math.factorial(k) )
        S_k = S * np.exp(k * np.log(m))
        sigma_k = np.sqrt( sigma**2 + k*(v**2) )
        r_k = (r - q) - lam*(m - 1)
        bs_price = put_BS(S_k, K, T, r_k, sigma_k, q=0.0)
        price += w_k * bs_price
    return price


# ------------- Implied-vol (Black–Scholes) from market prices ------------- #
# (Kept identical to your BS code so you retain the same UI/plots)
def impliedVol_call(p, S, K, T, r, q=0.0, max_iter=1000, tol=1e-6):
    """
    Black–Scholes implied volatility for a call using Newton–Raphson.
    """
    def f(sigma):
        return call_BS(S, K, T, r, sigma, q) - p
    
    def vega(sigma):
        if T <= 0:
            return 0
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
    
    sigma_guess = 0.3
    for _ in range(max_iter):
        diff = f(sigma_guess)
        if abs(diff) < tol:
            return sigma_guess
        v = vega(sigma_guess)
        if abs(v) < 1e-12:
            break
        sigma_guess -= diff / v
    return None

def impliedVol_put(p, S, K, T, r, q=0.0, max_iter=1000, tol=1e-6):
    """
    Black–Scholes implied volatility for a put using Newton–Raphson.
    """
    def f(sigma):
        return put_BS(S, K, T, r, sigma, q) - p
    
    def vega(sigma):
        if T <= 0:
            return 0
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
    
    sigma_guess = 0.3
    for _ in range(max_iter):
        diff = f(sigma_guess)
        if abs(diff) < tol:
            return sigma_guess
        v = vega(sigma_guess)
        if abs(v) < 1e-12:
            break
        sigma_guess -= diff / v
    return None


# ------------- Greeks via finite differences under MJD ------------- #
# For brevity, here we show Delta only, but you can repeat the pattern
# for Gamma, Vega, etc., if desired.
def delta_call_mjd(S, K, T, r, sigma, q, m, v, lam, h=1.0):
    """
    Finite-difference approximation to Delta of the MJD call.
    Uses price(S+ h) - price(S) as the difference.
    h=1.0 is large but matches your original approach.
    """
    p_up = merton_jump_call(S + h, K, T, r, sigma, q, m, v, lam)
    p_0  = merton_jump_call(S,     K, T, r, sigma, q, m, v, lam)
    return (p_up - p_0)/h

def delta_put_mjd(S, K, T, r, sigma, q, m, v, lam, h=1.0):
    p_up = merton_jump_put(S + h, K, T, r, sigma, q, m, v, lam)
    p_0  = merton_jump_put(S,     K, T, r, sigma, q, m, v, lam)
    return (p_up - p_0)/h


# ------------- Streamlit UI ------------- #

# Title and "Created by" area
col_i, col_t = st.columns([3,1])
with col_i:
    st.header("Merton Jump-Diffusion Option Pricing Model")
with col_t:
    st.markdown("""Created by 
        <a href="https://www.linkedin.com/in/remus-besliu/" target="_blank">
            <button style="background-color: #262730; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
                Remus Besliu
            </button>
        </a>
        """, unsafe_allow_html=True)


# Sidebar input parameters
st.sidebar.header('Input MJD parameters')
S = st.sidebar.number_input('Stock Price (S)', min_value=0.0, value=100.0, step=0.01)
K = st.sidebar.number_input('Strike Price (K)', min_value=0.0, value=120.0, step=0.01)
exp = st.sidebar.date_input('Expiry Date', value=dt.datetime(2025,9,19))
exp = dt.datetime.combine(exp, dt.datetime.min.time())
T = (exp - dt.datetime.today()).days / 365.0
r = st.sidebar.number_input('Risk Free Rate (r) in decimal', min_value=0.0, value=0.02, step=0.01)
sigma = st.sidebar.number_input('Volatility (σ) in decimal', min_value=0.0, value=0.2, step=0.01)
q = st.sidebar.number_input('Annual dividend yield (q)', min_value=0.0, value=0.0, step=0.01)

# Merton Jump Diffusion extras
m   = st.sidebar.number_input('Mean Jump Size (m)', min_value=0.01, value=1.0, step=0.01)
v   = st.sidebar.number_input('Jump Size Std Dev (v)', min_value=0.0, value=0.3, step=0.01)
lam = st.sidebar.number_input('Jump Intensity (λ)', min_value=0.0, value=1.0, step=0.1)


# Compute MJD call/put prices
callPrice = merton_jump_call(S, K, T, r, sigma, q, m, v, lam)
putPrice  = merton_jump_put(S, K, T, r, sigma, q, m, v, lam)

# Delta (example). If you like, define gamma, vega, etc. similarly.
deltaCall = delta_call_mjd(S, K, T, r, sigma, q, m, v, lam)
deltaPut  = delta_put_mjd(S, K, T, r, sigma, q, m, v, lam)

# Display results
col1, col2 = st.columns(2)
with col1:
    st.metric(label='MJD Call option price', value=f"${callPrice:.2f}")
with col2:
    st.metric(label='MJD Put option price',  value=f"${putPrice:.2f}")

with col1:
    with st.expander("Call Greeks (example)"):
        st.write(f"**Delta:** {deltaCall:.3f}")
        # If you implement them:
        # st.write(f"**Gamma:** ...")
        # st.write(f"**Vega:** ...")
        # st.write(f"**Theta:** ...")
        # st.write(f"**Rho:** ...")

with col2:
    with st.expander("Put Greeks (example)"):
        st.write(f"**Delta:** {deltaPut:.3f}")
        # If you implement them:
        # st.write(f"**Gamma:** ...")
        # st.write(f"**Vega:** ...")
        # st.write(f"**Theta:** ...")
        # st.write(f"**Rho:** ...")


# Heatmaps of MJD call/put as S and σ vary
S_values = np.linspace(S*1.5, S*0.5, 9)
sigma_values = np.linspace(0.1, 0.5, 9)
call_prices = np.zeros((len(S_values), len(sigma_values)))
put_prices  = np.zeros((len(S_values), len(sigma_values)))

for i, S_val in enumerate(S_values):
    for j, sigma_val in enumerate(sigma_values):
        call_prices[i, j] = merton_jump_call(S_val, K, T, r, sigma_val, q, m, v, lam)
        put_prices[i, j]  = merton_jump_put(S_val, K, T, r, sigma_val, q, m, v, lam)

call_fig = go.Figure(data=go.Heatmap(
    z=call_prices,
    x=np.round(sigma_values, 2),
    y=np.round(S_values, 2),
    colorscale='blues'))
call_fig.update_layout(
    title='Call Option Prices (MJD)',
    xaxis_title='Volatility (σ)',
    yaxis_title='Stock Price (S)')

put_fig = go.Figure(data=go.Heatmap(
    z=put_prices,
    x=np.round(sigma_values, 2),
    y=np.round(S_values, 2),
    colorscale='blues'))
put_fig.update_layout(
    title='Put Option Prices (MJD)',
    xaxis_title='Volatility (σ)',
    yaxis_title='Stock Price (S)')

colH1, colH2 = st.columns(2)
with colH1:
    st.plotly_chart(call_fig)
with colH2:
    st.plotly_chart(put_fig)


# Plot Greek sensitivities vs. Underlying Price
Stock_values = np.linspace(K*0.5, K*1.5, 100)
delta_call_vals = [delta_call_mjd(Sv, K, T, r, sigma, q, m, v, lam) for Sv in Stock_values]
# likewise gamma_call_vals, etc. if you implement them

fig_sens_call = go.Figure()
fig_sens_call.add_trace(go.Scatter(x=Stock_values, y=delta_call_vals,
                                   mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig_sens_call.update_layout(title='Call Greek Sensitivity vs. Underlying Price',
                            xaxis_title='Underlying Price',
                            yaxis_title='Value')

st.expander("Call Greek Sensitivity Graph").plotly_chart(fig_sens_call)

# Similarly for put
delta_put_vals = [delta_put_mjd(Sv, K, T, r, sigma, q, m, v, lam) for Sv in Stock_values]
fig_sens_put = go.Figure()
fig_sens_put.add_trace(go.Scatter(x=Stock_values, y=delta_put_vals,
                                  mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig_sens_put.update_layout(title='Put Greek Sensitivity vs. Underlying Price',
                           xaxis_title='Underlying Price',
                           yaxis_title='Value')

st.expander("Put Greek Sensitivity Graph").plotly_chart(fig_sens_put)


# Plot Greeks over time until expiration
# (For demonstration, just do Delta over time)
total_days = max((exp - dt.datetime.today()).days, 1)
exp_dates = pd.date_range(start=dt.datetime.today(), periods=total_days)
T_values = [(exp - d).days/365.0 for d in exp_dates]

delta_call_time = [delta_call_mjd(S, K, t, r, sigma, q, m, v, lam) for t in T_values]
delta_put_time  = [delta_put_mjd(S, K, t, r, sigma, q, m, v, lam)  for t in T_values]

fig_time_call = go.Figure()
fig_time_call.add_trace(go.Scatter(x=exp_dates, y=delta_call_time,
                                   mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig_time_call.update_layout(title='Call Delta Over Time Until Expiration',
                            xaxis_title='Date', yaxis_title='Delta')

fig_time_put = go.Figure()
fig_time_put.add_trace(go.Scatter(x=exp_dates, y=delta_put_time,
                                  mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig_time_put.update_layout(title='Put Delta Over Time Until Expiration',
                           xaxis_title='Date', yaxis_title='Delta')

colT1, colT2 = st.columns(2)
with colT1:
    st.expander("Call Greeks over time").plotly_chart(fig_time_call)
with colT2:
    st.expander("Put Greeks over time").plotly_chart(fig_time_put)


# Implied volatility section (still the Black–Scholes solver)
st.header("Implied Volatility (Black–Scholes style)")
col3, col4 = st.columns(2)
with col3:
    pcall = st.number_input('Market call option price', min_value=0.0, value=callPrice, step=0.01)
with col4:
    pput = st.number_input('Market put option price', min_value=0.0, value=putPrice, step=0.01)

IVcall = impliedVol_call(pcall, S, K, T, r, q)
IVput  = impliedVol_put(pput, S, K, T, r, q)

with col3:
    if IVcall is not None:
        st.metric('Implied volatility call', value=f"{IVcall:.2f}")
    else:
        st.write("No implied vol solution found for call.")

with col4:
    if IVput is not None:
        st.metric('Implied volatility put', value=f"{IVput:.2f}")
    else:
        st.write("No implied vol solution found for put.")


# Compare with historical volatility
st.header("Compare with historical volatility")
ticker = st.text_input('Yahoo Stock Ticker', value='AAPL')
col5, col6 = st.columns(2)
with col5:
    start = st.date_input('Start Date', value=dt.datetime(2021,1,1))
with col6:
    end   = st.date_input('End Date', value=dt.datetime.today())

if start < end:
    stockData = yf.download(ticker, start, end)
    stockData['dReturns'] = stockData['Close'].pct_change()
    stockData['HVol'] = stockData['dReturns'].rolling(window=30).std() * np.sqrt(252)

    fig_hv = go.Figure()
    fig_hv.add_trace(go.Scatter(
        x=stockData.index,
        y=stockData['HVol'],
        mode='lines',
        name='Historical Volatility (30d)',
        line=dict(color='lightblue')
    ))
    # Overplot the implied vol lines
    if IVcall is not None:
        fig_hv.add_trace(go.Scatter(
            x=[stockData.index.min(), stockData.index.max()],
            y=[IVcall, IVcall],
            mode='lines',
            name='Implied Vol Call',
            line=dict(color='blue')
        ))
    if IVput is not None:
        fig_hv.add_trace(go.Scatter(
            x=[stockData.index.min(), stockData.index.max()],
            y=[IVput, IVput],
            mode='lines',
            name='Implied Vol Put',
            line=dict(color='purple')
        ))

    fig_hv.update_layout(
        title='Historical and Implied Volatility',
        xaxis_title='Date',
        yaxis_title='Volatility',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0.75)
    )
    st.plotly_chart(fig_hv, use_container_width=True)
else:
    st.write("Error: Start date must be before End date!")