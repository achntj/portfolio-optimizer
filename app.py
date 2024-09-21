import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimization_functions import markowitz_optimization, monte_carlo_simulation, plot_efficient_frontier, find_best_portfolio, calculate_cvar, display_optimized_portfolio


import yfinance as yf


st.title('Portfolio Optimization Tool')
st.write("This tool uses Modern Portfolio Theory - specifically Markowitz Optimization Theory to find the optimal portfolio at your chosen target returns. It also runs monte carlo simulations with 5000 portfolios to find the best portfolio (with highest Sharpe Ratio)")


def get_asset_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

tickers = st.text_input("Enter Tickers from your portfolio", "AAPL, TXNM, MSFT, JNJ, CL, NVDA")
tickers  = [ticker.strip() for ticker in tickers.split(",")]
start_date = '2020-01-01'
end_date = '2023-01-01'

data = get_asset_data(tickers, start_date, end_date)
data.dropna(inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))
data.plot(ax=ax)
plt.title("Stock Prices")
st.pyplot(fig)

returns = data.pct_change().dropna()
expected_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252


target_pct  = st.slider("Target Return %", 0, 100)
target_return = target_pct / 100
optimal_weights = markowitz_optimization(expected_returns, cov_matrix, target_return)

st.divider()

st.header("Best Portfolio Weights: ")
# Run Monte Carlo Simulation and plot Efficient Frontier
simulation_results, weights_record = monte_carlo_simulation(expected_returns, cov_matrix)
# Find the best portfolio from the simulation results
best_portfolio = find_best_portfolio(simulation_results, weights_record)
plot_efficient_frontier(simulation_results)
st.pyplot(plt.gcf()) # instead of plt.show()

portfolio_df = pd.DataFrame({'Asset': tickers, 'Optimal Weight': best_portfolio['weights']})
st.dataframe(portfolio_df)
st.text(f"Expected Return: {best_portfolio['return']:.2%}")
st.text(f"Risk : {best_portfolio['risk']:.2%}")
st.text(f"Sharpe Ratio: {best_portfolio['sharpe_ratio']:.2f}")


display_optimized_portfolio(tickers, expected_returns, cov_matrix, target_return, simulation_results)
cvar, var = calculate_cvar(returns, optimal_weights, confidence_level=0.95)
st.text(f"Value at Risk (VaR) at 95% confidence level: {var:.2%}")
st.text(f"Conditional Value at Risk (CVaR) at 95% confidence level: {cvar:.2%}")
