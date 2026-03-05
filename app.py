import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ---------------------------
# BLACK-SCHOLES CORE
# ---------------------------

class BlackScholes:
    """
    Black-Scholes option pricing model for European options

    Parameters
    ----------
    S     : float   Current stock price
    K     : float   Strike price
    T     : float   Time to expiry (in years)
    r     : float   Risk-free rate (as decimal, e.g. 0.05 for 5%)
    sigma : float   Volatility (as decimal, e.g. 0.20 for 20%)
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self) -> float:
        return (np.log(self.S / self.K) +
                (self.r + 0.5 * self.sigma ** 2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    # Here norm.cdf(d1) = delta. (lognormal distribution)
    # "Distance" of stock from strike in volatility units. Drives delta and most Greeks


    def d2(self) -> float:
        return self.d1() - self.sigma * np.sqrt(self.T)
    # Similar to d1 but shifted forward in time; tied to probability of finishing ITM.

    def call_price(self) -> float:
        return (self.S * norm.cdf(self.d1()) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
    # Expected value of exercising the call under risk-neutral pricing

    def put_price(self) -> float:
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) -
                self.S * norm.cdf(-self.d1()))
    # Same idea as call but for downside protection.

    # ---------------------------
    # GREEKS
    # ---------------------------

    def delta_call(self) -> float:
        """Rate of change of option price per $1 move in stock."""
        return norm.cdf(self.d1())
    # How much the option behaves like stock, approximately the probability call finishes ITM

    def delta_put(self) -> float:
        return norm.cdf(self.d1()) - 1
    # Negative stock exposure; put gains when stock falls


    def gamma(self) -> float:
        """Rate of change of delta per $1 move in stock. Same for calls and puts."""
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    # # How quickly delta changes as price moves (curvature of the option).

    def vega(self) -> float:
        """
        Change in option price per 1% move in implied volatility.
        Divided by 100 to convert from per-unit to per-percentage-point.
        """
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100
    # Essential sensitivity to volatility, options will get more valuable when uncertainty rises


    def theta_call(self) -> float:
        """
        Daily time decay for a call option.
        Divided by 365 to convert from annual to daily decay.
        """
        annual = (-self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T))
                  - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
        return annual / 365
    # Value lost each day as expiration approaches ("time decay")


    def theta_put(self) -> float:
        """
        Daily time decay for a put option.
        Divided by 365 to convert from annual to daily decay.
        """
        annual = (-self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T))
                  + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))
        return annual / 365
    # Same time decay effect but for puts.

    def rho_call(self) -> float:
        """Change in option price per 1% move in risk-free rate."""
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2()) / 100
    # Sensitivity to interest rates (calls usually rise when rates rise

    def rho_put(self) -> float:
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100
    # Opposite rate effect: puts usually fall when rates rise

    # ---------------------------
    # IMPLIED VOLATILITY SOLVER
    # --------------------------

    # # Newton Raphson method used to solve for implied volatility
    # We will iteratively adjust sigma until themodel price = market price

    @staticmethod
    def implied_volatility(S: float, K: float, T: float, r: float,
                           market_price: float, option_type: str = "call",
                           tol: float = 1e-6, max_iterations: int = 100) -> float:
        """
        Newton-Raphson solver for implied volatility.

        Iteratively adjusts sigma until the model price matches the observed
        market price within the specified tolerance.

        IV is the market's consensus estimate of future volatility, implied
        by the price traders are actually willing to pay.
        """
        sigma = 0.2  # initial guess: 20% volatility

        for _ in range(max_iterations):
            bs = BlackScholes(S, K, T, r, sigma)
            price = bs.call_price() if option_type == "call" else bs.put_price()
            vega = bs.vega() * 100  # use raw vega (not per-%) for Newton step

            price_diff = price - market_price
            # difference between model price and actual market option price


            if abs(price_diff) < tol:
                return sigma
            #if model price is close enough to market price, we found implied volatility


            if vega < 1e-10:  # avoid division by zero near expiry
                break
            # if the vega is too small the solver becomes unstable (because option barely reacts to volatility)


            sigma -= price_diff / vega
            # Newton-Raphson update, adjust volatility to reduce pricing error


            if sigma <= 0:  # sigma must stay positive
                sigma = 1e-6
            # volatility cannot be negative, clamp to small positive value


        return sigma
    # return the best estimate we have if solver doesn't fully converge


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="Black-Scholes Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Black-Scholes Options Pricing & Greeks Dashboard")
st.markdown(
    "European options pricing model with full Greeks and implied volatility solver. "
    "Built from first principles using NumPy and SciPy."
)

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------

st.sidebar.header("Model Parameters")

S     = st.sidebar.slider("Stock Price (S)",        10.0,  300.0, 100.0, step=1.0)
K     = st.sidebar.slider("Strike Price (K)",       10.0,  300.0, 100.0, step=1.0)
T     = st.sidebar.slider("Time to Expiry (Years)", 0.01,    2.0,   1.0, step=0.01)
r     = st.sidebar.slider("Risk-Free Rate (%)",      0.0,   10.0,   5.0, step=0.1) / 100
sigma = st.sidebar.slider("Volatility (%)",          1.0,  100.0,  20.0, step=0.5) / 100

st.sidebar.markdown("---")
st.sidebar.header("Implied Volatility Solver")

market_price = st.sidebar.number_input(
    "Observed Market Option Price ($)",
    min_value=0.01,
    value=10.0,
    step=0.01
)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# ---------------------------
# MODEL COMPUTATION
# ---------------------------

bs = BlackScholes(S, K, T, r, sigma)

call_price = bs.call_price()
put_price  = bs.put_price()

iv = BlackScholes.implied_volatility(S, K, T, r, market_price, option_type)

# ---------------------------
# PRICES + IV DISPLAY
# ---------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Call Price",          f"${call_price:.4f}")
col2.metric("Put Price",           f"${put_price:.4f}")
col3.metric("Implied Volatility",  f"{iv * 100:.2f}%",
            help="Market-implied volatility solved via Newton-Raphson from the observed option price.")

# -------------------------
# GREEKS TABLE
# --------------------------

st.subheader("Greeks")
st.caption("Vega = per 1% change in vol | Theta = daily decay | Rho = per 1% change in rate")

greeks_df = pd.DataFrame({
    "Greek":       ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
    "Description": [
        "Price change per $1 stock move",
        "Delta change per $1 stock move",
        "Price change per 1% vol move",
        "Daily time decay ($)",
        "Price change per 1% rate move"
    ],
    "Call": [
        round(bs.delta_call(), 4),
        round(bs.gamma(),      4),
        round(bs.vega(),       4),
        round(bs.theta_call(), 4),
        round(bs.rho_call(),   4),
    ],
    "Put": [
        round(bs.delta_put(),  4),
        round(bs.gamma(),      4),
        round(bs.vega(),       4),
        round(bs.theta_put(),  4),
        round(bs.rho_put(),    4),
    ]
})

st.dataframe(greeks_df, use_container_width=True, hide_index=True)

# -------------------------
# PRICE + GREEKS CURVES
# ------------------------

S_range      = np.linspace(0.5 * S, 1.5 * S, 200)
call_prices  = []
put_prices   = []
delta_calls  = []
delta_puts   = []
gamma_vals   = []
vega_vals    = []
theta_calls  = []

for s in S_range:
    b = BlackScholes(s, K, T, r, sigma)
    call_prices.append(b.call_price())
    put_prices.append(b.put_price())
    delta_calls.append(b.delta_call())
    delta_puts.append(b.delta_put())
    gamma_vals.append(b.gamma())
    vega_vals.append(b.vega())
    theta_calls.append(b.theta_call())

# Call and Put Price formatting
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=S_range, y=call_prices, name="Call Price",
                               line=dict(color="#2563eb", width=2)))
fig_price.add_trace(go.Scatter(x=S_range, y=put_prices,  name="Put Price",
                               line=dict(color="#dc2626", width=2)))
fig_price.add_vline(x=S, line_dash="dash", line_color="gray",
                    annotation_text=f"Current S = {S}")
fig_price.update_layout(title="Option Price vs Stock Price",
                        xaxis_title="Stock Price ($)",
                        yaxis_title="Option Price ($)",
                        hovermode="x unified")
st.plotly_chart(fig_price, use_container_width=True)

# Delta 
fig_delta = go.Figure()
fig_delta.add_trace(go.Scatter(x=S_range, y=delta_calls, name="Call Delta",
                               line=dict(color="#2563eb", width=2)))
fig_delta.add_trace(go.Scatter(x=S_range, y=delta_puts,  name="Put Delta",
                               line=dict(color="#dc2626", width=2)))
fig_delta.add_vline(x=S, line_dash="dash", line_color="gray",
                    annotation_text=f"Current S = {S}")
fig_delta.update_layout(title="Delta vs Stock Price — Hedge Ratio",
                        xaxis_title="Stock Price ($)",
                        yaxis_title="Delta",
                        hovermode="x unified")
st.plotly_chart(fig_delta, use_container_width=True)

# Gamma
fig_gamma = go.Figure()
fig_gamma.add_trace(go.Scatter(x=S_range, y=gamma_vals, name="Gamma",
                               line=dict(color="#16a34a", width=2),
                               fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"))
fig_gamma.add_vline(x=S, line_dash="dash", line_color="gray",
                    annotation_text=f"Current S = {S}")
fig_gamma.update_layout(title="Gamma vs Stock Price — Convexity",
                        xaxis_title="Stock Price ($)",
                        yaxis_title="Gamma",
                        hovermode="x unified")
st.plotly_chart(fig_gamma, use_container_width=True)

# --------------------------
# OPTION PRICE HEATMAP
# --------------------------

st.subheader("Option Price Surface — Stock Price × Volatility")
st.caption("Full pricing surface across stock price and volatility. "
           "This is what quant desks visualise when stress-testing positions!!")

S_heat     = np.linspace(0.6 * S, 1.4 * S, 60)
sigma_heat = np.linspace(0.05, 0.80, 60)
# range of stock prices around the current price
#range of volatility levels (5% to 80%)



call_surface = np.array([
    [BlackScholes(s, K, T, r, sig).call_price() for s in S_heat]
    for sig in sigma_heat
])
# compute call prices for every combination of stock price and volatility

put_surface = np.array([
    [BlackScholes(s, K, T, r, sig).put_price() for s in S_heat]
    for sig in sigma_heat
])
# same idea but for put options

heat_tab1, heat_tab2 = st.tabs(["Call Surface", "Put Surface"])
# create 2 tabs so users can switch between call and put visualisations

with heat_tab1:
    fig_heatmap_call = go.Figure(data=go.Heatmap(
        z=call_surface,
        x=np.round(S_heat, 1),
        y=np.round(sigma_heat * 100, 1),
        colorscale="Blues",
        colorbar=dict(title="Call Price ($)")
    ))
    # heatmap showing how call price changes with stock price and volatility
    fig_heatmap_call.update_layout(
        title="Call Price Surface",
        xaxis_title="Stock Price ($)",
        yaxis_title="Volatility (%)"
    )
    # label axes so the chart is easy to interpret
    st.plotly_chart(fig_heatmap_call, use_container_width=True)
    # render the interactive heatmap in the Streamlit app

with heat_tab2:
    fig_heatmap_put = go.Figure(data=go.Heatmap(
        z=put_surface,
        x=np.round(S_heat, 1),
        y=np.round(sigma_heat * 100, 1),
        colorscale="Reds",
        colorbar=dict(title="Put Price ($)")
    ))
    # heatmap showing how put price changes with stock price and volatility
    fig_heatmap_put.update_layout(
        title="Put Price Surface",
        xaxis_title="Stock Price ($)",
        yaxis_title="Volatility (%)"
    )
    st.plotly_chart(fig_heatmap_put, use_container_width=True)

# ---------------------------
# IMPLIED VOLATILITY SMILE
# ---------------------------

# # Simulated volatility smile :) 
# In real markets OTM puts have higher IV (downside crash protection demand)

st.subheader("Implied Volatility Smile")
st.caption(
    "In a perfect Black-Scholes world, IV would be flat across all strikes. "
    "In reality, markets price OTM puts at a premium — known as the volatility skew or smile. "
    "This simulation adds a realistic skew manually since Black-Scholes itself cannot generate it."
)

K_range  = np.linspace(0.70 * S, 1.30 * S, 60)
# create a range of strike prices around the current stock price

iv_smile = []
# create a range of strike prices around the current stock price


for strike in K_range:
    # Simulate realistic market skew:
    # OTM puts (low strike) trade at higher IV — crash protection premium
    # ATM options trade near model vol
    # OTM calls trade slightly above model vol
    moneyness = (S - strike) / S # create a range of strike prices around the current stock price
    skew      = 0.30 * moneyness          # downside skew
    curvature = 0.15 * moneyness ** 2     # smile curvature, quadratic term creates the "smile" shape around ATM
    simulated_iv = sigma + skew + curvature # adjust base volatility using skew and curvature
    simulated_iv = max(simulated_iv, 0.01)  # floor at 1% ensure volatility never goes below 1%
    iv_smile.append(simulated_iv * 100)     # store IV as percentage for plotting

fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(
    x=K_range, y=iv_smile,
    name="Implied Volatility",
    line=dict(color="#7c3aed", width=2),
    fill="tozeroy", fillcolor="rgba(124,58,237,0.06)"
))
fig_smile.add_vline(x=S, line_dash="dash", line_color="gray",
                    annotation_text="ATM")
#vertical line showing the at-the-money strike
fig_smile.update_layout(
    title="Implied Volatility Smile / Skew",
    xaxis_title="Strike Price ($)",
    yaxis_title="Implied Volatility (%)",
    hovermode="x unified"
)
st.plotly_chart(fig_smile, use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")
st.markdown("""
**Black-Scholes Options Pricing Dashboard**

Implements the Black-Scholes model for European options from first principles.

**Model features:**
- Closed-form call and put pricing
- Full Greeks: Δ Delta, Γ Gamma, ν Vega (per 1% vol), Θ Theta (daily), ρ Rho
- Newton-Raphson implied volatility solver
- Option price surface heatmap across stock price × volatility
- Simulated volatility smile with realistic downside skew

**Key model assumptions:**
- Constant volatility (real markets violate this, hence the smile)
- Lognormal returns (real returns have fat tails)
- Continuous trading and no transaction costs
- European-style exercise only

Built using Python · NumPy · SciPy · Plotly · Streamlit
""")