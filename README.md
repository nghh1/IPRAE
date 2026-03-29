# Investment Portfolio Risk Analyser
An quantitative risk instrument designed to calculate Value at Risk (VaR) and evaluate portfolio resilience through Multivariate Monte Carlo Simulations and systemic stress testing.

## Project Overview
This analyser enables investors to move beyond static historical analysis. By simulating thousands of correlated market paths, it identifies market crash vulnerabilities and suggests data-driven hedging strategies to protect capital during market panics.

## Key Features
- Multivariate Monte Carlo Method: Simulates throusands of price paths and computes logarithmic returns and covariance matrices to preserve asset relationships.
- Systemic Stress Testing: A stress test module that applies volatility multipliers, overnight gaps, and negative drift to model liquidity contagion.
- Institutional Analytics: Real-time calculation of 95% VaR and Sharpe Ratios (risk-adjusted returns).
- Interactive Dashboard: A Streamlit UI interface demonstrating historical asset performance, asset correlation heatmap, and hedging strategy.
- Automated Data Pipeline: Dynamic ingestion of global market data via the Yahoo Finance API.

## Getting Started

### Option 1: Using Docker

**Build a Docker container image and run**
```bash
docker build -t iprae .
docker run -p 8501:8501 iprae
```

### Option 2: Using uv

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Automatically install all required packages from the lockfile
uv sync

# Run the Streamlit dashboard
streamlit run dashboard.py
```

## Tech Stack
- Language: Python 3.12
- Data manipulation: Pandas, NumPy, yfinance
- Visualization: Plotly, Seaborn, Matplotlib
- DevOps: Docker, uv
- UI: Streamlit
