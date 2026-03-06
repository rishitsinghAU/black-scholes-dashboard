# Black-Scholes Options Pricing Dashboard

This is an interactive options pricing dashboard built from first principles, implementing the Black-Scholes model for European options with full Greeks computation and a Newton-Raphson implied volatility solver!

Built with Python · NumPy · SciPy · Plotly · Streamlit

What Does It Do?

- Prices European call and put options** using the closed-form Black-Scholes solution
- Computes all five Greeks with correct real-world units (Theta in daily decay, Vega per 1% vol move)
- Solves for implied volatility from an observed market price using Newton-Raphson iteration
- Visualises the full option pricing surface** across stock price × volatility as a heatmap
- Displays a realistic volatility smile** with downside skew, illustrating where Black-Scholes breaks down in real markets

## The Model

At the core is the Black-Scholes PDE, which assumes stock prices follow geometric Brownian motion:

```
dS = μS dt + σS dW
```

$$
dS = \mu S\;dt+\sigma S\;dW
$$

This leads to the closed-form pricing formulas:

```
Call = S·N(d₁) - K·e^(-rT)·N(d₂)
Put  = K·e^(-rT)·N(-d₂) - S·N(-d₁)

d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
d₂ = d₁ - σ√T
```

Where `N(·)` is the cumulative standard normal distribution.

## Greeks

 Delta (Δ)  Option price change per $1 stock move  $ per $ 
 Gamma (Γ)  Delta change per $1 stock move | Convexity |
 Vega (ν)  Option price change per 1% vol move  $ per 1% σ 
 Theta (Θ)  Time decay | $ per day |
 Rho (ρ)  Option price change per 1% rate move  $ per 1% r 

Vega is divided by 100 (sensitivity per percentage point, not per unit) and Theta is divided by 365 (daily decay, not annual). These match the typical industry convention!

## Implied Volatility Solver

Rather than plugging in volatility to get a price, real markets work in reverse, in that traders will observe a market price and solve backwards for the volatility that produced it. This is the implied volatility (IV).

The solver uses Newton-Raphson iteration:

```
σ_new = σ_old - (model_price - market_price) / vega
```

Each step adjusts the volatility estimate by the pricing error divided by Vega (the slope of the price-vol relationship), converging to the solution within microseconds!

## Volatility Smile

In a perfect Black-Scholes world, IV would be flat across all strikes. In reality however it is not, out-of-the-money puts trade at elevated IV because investors pay a **premium** for crash protection. This creates the characteristic volatility skew (sometimes called the smile :).

The dashboard simulates a realistic skew since Black-Scholes itself cannot generate one, this is one of the model's known structural limitations.

## Model Assumptions & Limitations

Black-Scholes makes several simplifying assumptions that real markets routinely violate:

- Constant volatility real volatility is stochastic (hence the smile)
- Lognormal returns real returns have fat tails; crashes happen more often than the model predicts
- Continuous trading not possible in practice
- No transaction costs not realistic
- European exercise only cannot price American options which can be exercised early

These limitations are features. Understanding where the model breaks down will be the foundation of more advanced volatility modelling (stochastic vol models, local vol surfaces, and more)

---

# Installation

```bash
git clone https://github.com/rishitsinghAU/black-scholes-dashboard
cd black-scholes-dashboard
pip install -r requirements.txt
streamlit run app.py
```

### requirements.txt**
```
numpy
pandas
scipy
streamlit
plotly
```

---

## Project Structure

```
app.py   # Full model, Greeks, IV solver, Streamlit dashboard
README.md
requirements.txt
```

### Learning Notes

This project was built as a personal learning exercise to better understand the Black-Scholes model, option Greeks, and implied volatility. Some concepts were learned from lectures, textbooks, and online resources while building the dashboard.

The goal of the project was to implement the model from first principles and develop intuition for how option prices respond to changes in stock price, volatility, and time.
