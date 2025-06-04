import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go

def black_scholes_call_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    put_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

    return {
        'call_price': call_price,
        'put_price': put_price,
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'vega': vega,
        'theta_call': theta_call,
        'theta_put': theta_put
    }

S = 100
K = 100
T = 0.5
r = 0.05
sigma = 0.2

results = black_scholes_call_put(S, K, T, r, sigma)
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Payoff graph
S_range = np.linspace(50, 150, 200)
call_payoff = np.maximum(S_range - K, 0) - results['call_price']
put_payoff = np.maximum(K - S_range, 0) - results['put_price']

fig = go.Figure()
fig.add_trace(go.Scatter(x=S_range, y=call_payoff, name="Call Payoff", line=dict(color='green')))
fig.add_trace(go.Scatter(x=S_range, y=put_payoff, name="Put Payoff", line=dict(color='red')))
fig.add_trace(go.Scatter(x=S_range, y=np.zeros_like(S_range), name="Break-even", line=dict(color='black', dash='dash')))

fig.update_layout(title="Payoff à maturité - Call et Put",
                  xaxis_title="Prix du sous-jacent à maturité (S)",
                  yaxis_title="Profit / Perte",
                  template="plotly_white")
fig.show()
