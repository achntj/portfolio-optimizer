# Portfolio Optimization Web App

This project is a web application built to optimize investment portfolios using **Markowitz Portfolio Theory**, **Black-Litterman Model**, and **Monte Carlo Simulations**. The web app allows users to adjust your target returns, create a portfolio and immediately see the effects on expected returns and risk through real-time visualizations.

### Price Simulation (GBM)

Simulates future asset prices using the Geometric Brownian Motion (GBM) process, incorporating the expected returns and volatilities of each asset. The simulation allows us to evaluate how the portfolio might perform over a specified time period.

### **Risk Assessment Techniques**

The app also includes a few core risk assessment methodologies:

- **Portfolio Variance & Standard Deviation:** These metrics are used to quantify the overall portfolio risk. The covariance matrix is key to measuring how individual assets' risks combine.
- **Value at Risk (VaR):** VaR measures the potential loss in value of an asset or portfolio over a defined period for a given confidence interval.
- **Conditional Value at Risk (CVaR):** This is an extension of VaR that measures the expected loss, assuming that the loss has exceeded the VaR threshold.
