# Portfolio Optimization Web App

This project is a web application built to optimize investment portfolios using **Markowitz Portfolio Theory**, **Black-Litterman Model**, and **Monte Carlo Simulations**. The web app allows users to adjust your target returns, create a portfolio and immediately see the effects on expected returns and risk through real-time visualizations.

## Models and Techniques Used

### 1. **Markowitz's Model**

Markowitz’s Model is a mathematical framework for assembling a portfolio of assets to maximize expected return for a given level of risk. It assumes that investors are risk-averse, meaning they prefer a portfolio with the lowest possible risk for a given return. The model uses:

- **Expected Return:** The weighted sum of the expected returns of the assets in the portfolio.
- **Covariance Matrix:** This measures how asset returns move in relation to each other, allowing for the calculation of the overall portfolio risk.
- **Optimization Objective:** Minimize the variance (risk) while maximizing returns, based on user-defined portfolio weights.

### 2. **Black-Litterman Model**

The Black-Litterman Model extends the Markowitz framework by incorporating subjective views of the investor. It adjusts the expected returns based on market equilibrium and the investor’s personal opinions about asset performance. The key features include:

- **Equilibrium Return:** Reflects the expected return based on the global market.
- **Investor Views:** Allows the investor to incorporate their beliefs into the asset return expectations, which modifies the portfolio optimization process.

### 3. **Risk Assessment Techniques**

The app also includes a few core risk assessment methodologies:

- **Portfolio Variance & Standard Deviation:** These metrics are used to quantify the overall portfolio risk. The covariance matrix is key to measuring how individual assets' risks combine.
- **Value at Risk (VaR):** VaR measures the potential loss in value of an asset or portfolio over a defined period for a given confidence interval.
- **Conditional Value at Risk (CVaR):** This is an extension of VaR that measures the expected loss, assuming that the loss has exceeded the VaR threshold.

### 4. **Monte Carlo Simulations**

Monte Carlo simulation is used to generate a wide range of potential portfolio outcomes by simulating random returns for each asset. This allows for a better understanding of the distribution of potential risks and returns, enabling more robust decision-making.

## Visualization and Plots

The app features interactive visualizations that help users understand their portfolio performance:

1. **Portfolio Weights Bar Chart:**
   A bar chart displays the user-selected asset weights in the portfolio. This helps users visually adjust and optimize their portfolios.

2. **Stock Price Chart (Optional):**
   If stock price data is available, the app can generate a time series plot showing historical stock prices. This helps users understand asset performance over time.

3. **Risk and Return Metrics:**
   After adjusting the portfolio weights, users can instantly see the expected return and the portfolio's standard deviation (risk). This helps them understand the trade-off between risk and return.
