# Investment Portfolio Risk Analysis Engine 
An quantitative risk instrument designed to calculate Value at Risk (VaR) and evaluate portfolio resilience through Multivariate Monte Carlo Simulations and systemic stress testing.

## Project Overview
This engine enables investors to move beyond static historical analysis. By simulating thousands of correlated market paths, it identifies market crash vulnerabilities and suggests data-driven hedging strategies to protect capital during market panics.

## Key Features
- Multivariate Monte Carlo Engine: Simulates throusands of price paths and computes logarithmic returns and covariance matrices to preserve asset relationships.
- Systemic Stress Testing: A stress test module that applies volatility multipliers, overnight gaps, and negative drift to model liquidity contagion.
- Institutional Analytics: Real-time calculation of 95% VaR and Sharpe Ratios (risk-adjusted returns).
- Interactive Dashboard: A Streamlit UI interface demonstrating historical asset performance, asset correlation heatmap, and hedging strategy.
- Automated Data Pipeline: Dynamic ingestion of global market data via the Yahoo Finance API.

## Getting Started

### Build the docker image
`docker build -t iprae .`
### Run the container
`docker run -p 8501:8501 iprae`

## Tech Stack
- Language: Python 3.11
- Data: Pandas, NumPy, yfinance
- Visualization: Plotly, Seaborn, Matplotlib
- DevOps: Docker, uv
- UI: Streamlit