from cvxopt import matrix, solvers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def markowitz_optimization(expected_returns, cov_matrix, target_return):
    n = len(expected_returns)
    P = matrix(cov_matrix.values)
    q = matrix(np.zeros((n, 1)))
    G = matrix(-np.identity(n))
    h = matrix(np.zeros((n, 1)))
    A = matrix(np.vstack((expected_returns.values, np.ones(n))))
    b = matrix([target_return, 1.0])
    sol = solvers.qp(P, q, G, h, A, b)
    if sol['status'] != 'optimal':
        st.text("Optimization failed with status: " + sol['status'])
        return None
    weights = np.array(sol['x']).flatten()
    return weights

def monte_carlo_simulation(expected_returns, cov_matrix, num_portfolios=50000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    n_assets = len(expected_returns)
    for i in range(num_portfolios):
        weights = np.random.rand(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk
        results[0, i] = portfolio_return
        results[1, i] = portfolio_risk
        results[2, i] = sharpe_ratio
    return results, weights_record

def plot_efficient_frontier(results):
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='.', s=1)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.show()

def find_best_portfolio(results, weights_record):
    max_sharpe_idx = np.argmax(results[2])  # Index of portfolio with max Sharpe ratio
    max_sharpe_weights = weights_record[max_sharpe_idx]
    
    return {
        'weights': max_sharpe_weights,
        'return': results[0, max_sharpe_idx],
        'risk': results[1, max_sharpe_idx],
        'sharpe_ratio': results[2, max_sharpe_idx]
    }

def display_optimized_portfolio(tickers, expected_returns, cov_matrix, target_return, simulation_results):
    optimal_weights = markowitz_optimization(expected_returns, cov_matrix, target_return)
    if optimal_weights is None:
        st.text("Optimization did not find a solution.")
        return
    portfolio_df = pd.DataFrame({'Asset': tickers, 'Optimal Weight': optimal_weights})
    st.header("Optimized Portfolio Allocation:\n")
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    # Plot the optimized portfolio on the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(simulation_results[1, :], simulation_results[0, :], c=simulation_results[2, :], cmap='viridis', marker='.', s=1, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(portfolio_risk, portfolio_return, color='red', marker='*', s=300, label='Optimized Portfolio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier with Optimized Portfolio')
    plt.legend()
    plt.show()
    st.pyplot(plt.gcf()) # instead of plt.show()
    st.dataframe(portfolio_df)
    st.text(f"\nExpected Portfolio Return: {portfolio_return:.2%}")
    st.text(f"Expected Portfolio Risk (Volatility): {portfolio_risk:.2%}")


def calculate_cvar(returns, weights, confidence_level=0.95):
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Sort portfolio returns in ascending order
    sorted_returns = np.sort(portfolio_returns)
    
    # Calculate the index of the VaR threshold
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    # Calculate the VaR
    var = sorted_returns[var_index]
    
    # Calculate the CVaR (average of returns below the VaR threshold)
    cvar = sorted_returns[:var_index].mean()
    
    return cvar, var

