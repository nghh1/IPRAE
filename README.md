# Investment Portfolio Risk Analyser
This is a risk analyser built to stress-test financial portfolios. It uses non-parametric historical bootstrapping to simulate thousands potential future growth paths, quantifying potential losses through risk metrics.

## Key Features
- Dynamic Simulation Engine:

    - Historical Bootstrapping: Randomly samples historical log-returns to preserve the unique volatility of your assets.

    - Buy & Hold vs. Rebalancing: Toggle between a passive strategy (where weights drift based on performance) and an active strategy (daily rebalancing to target weights).

- Systemic Stress Testing:

    - Volatility Shocks: Amplify historical swings (e.g., 2x or 3x market panic).

    - Market Crash Events: Model overnight gaps (liquidity crises) and sustained negative drift (secular bear markets).

- Advanced Risk Analytics:

    - VaR & CVaR: Measures the minimum and average loss in the worst 5% of outcomes.

    - Realised Sharpe Ratio: Calculates risk-adjusted returns based on simulated path outcomes rather than static historical averages.

    - Risk Attribution: Identifies which specific asset is hogging the portfolio's risk budget.

    - Diversification Ratio: A mathematical score of how well your assets offset each other's risks.

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
