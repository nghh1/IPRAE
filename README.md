# Investment Portfolio Risk Analyser Engine (IPRAE)

A production-grade, containerised quantitative risk engine that performs Monte Carlo stress testing on custom investment portfolios.

Designed with a high-level mathematical backend and a decoupled microservice architecture, IPRAE accurately forecasts Tail Risk (CVaR), Value at Risk (VaR), and risk-adjusted returns (Sortino) under both normal and severe market crash conditions.

## Project Structure

```Paintext
IPRAE/
├── .streamlit/
│   └── secrets.toml                   # API keys (Git-ignored)
├── api.py                             # FastAPI Backend application
├── dashboard.py                       # Streamlit Frontend UI
├── DataPipeline.py                    # Alpaca/YFinance data ingestion layer
├── docker-compose.yml                 # Docker cluster orchestrator
├── Dockerfile                         # Container image builder 
├── InvestmentPortfolioStressTester.py # Core Monte Carlo math engine
├── pyproject.toml                     # uv project definitions
└── uv.lock                            # Deterministic dependency tree
```

## Key Features

### Microservice Architecture
* **Frontend (Streamlit):** A lightweight, reactive client that caches data and handles UI rendering.
* **Backend (FastAPI):** A heavy-duty, asynchronous calculation engine strictly dedicated to processing Monte Carlo matrix math.
* **Resilient Data Pipeline:** Integrates with **Alpaca API** for premium institutional data, with a graceful, automatic fallback to **Yahoo Finance** if keys are missing.

### Modern DevOps
* **Package Manager:** Uses `uv` for dependency resolution.
* **Dockerised Cluster:** A mono-repo Docker Compose setup with strict volume mapping and isolated virtual environments for seamless deployment.

## Architecture Overview

The application is split into two distinct containers communicating over an internal Docker network:

1. **User configures portfolio** via the `Streamlit` UI.
2. **DataPipeline** checks for `secrets.toml`. If Alpaca keys are present, it fetches timezone-aware premium data. If not, it falls back to `yfinance`.
3. The UI caches the clean dataset and sends it along with the user's stress parameters as a JSON payload to the `FastAPI` backend.
4. The **Backend** processes the Monte Carlo simulations (up to 3,000 paths) and returns the projected bounds, CVaR, and correlation matrices.
5. The **Frontend** decodes the JSON and renders interactive `Plotly` and `Seaborn` charts.

## Getting Started

The easiest and most reliable way to run this application is via Docker.

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 2. Configure API Secrets
Create a `.streamlit` folder in the project root and add a `secrets.toml` file.

```bash
mkdir .streamlit
touch .streamlit/secrets.toml
```

Add your Alpaca API credentials (Paper Trading keys). Note: If you leave this file empty or skip this step, the application will fall back to Yahoo Finance API.

```toml
# .streamlit/secrets.toml
[api_keys]
ALPACA_API_KEY = "your_alpaca_api_key"
ALPACA_API_SECRET = "your_alpaca_api_secret"
```

### 3. Launch the Cluster
Start the services using Docker Compose

```bash
docker-compose up --build
```

* **Frontend UI:** Open your brower to `http://localhost:8501`
* **Backend API Docs:** Open your brower to `http://localhost:8000/docs`

## Local Testing (Without Docker)

If you wish employ locally without containers, you must use `uv` to manage the environment.

### 1. Install Dependencies
```bash
# Install uv if you haven't already
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
# Sync the virtual environment
uv sync
```

### 2. Start the Backend (Terminal 1)
```bash
uv run uvicorn api:app --reload
```

### 3. Start the Frontend (Terminal 2)
```bash
uv run streamlit run dashboard.py
```

## Tech Stack

- Frontend & Visualisation
  - Streamlit: Acts as the reactive frontend framework. It handles the user inputs, caches the raw data to memory, and renders the dashboard.

  - Plotly: Used for rendering the highly interactive, responsive charts (e.g., the Base-100 historical trends and the Monte Carlo cone charts).

  - Seaborn & Matplotlib: Used specifically to generate the static statistical visualisations, like the Asset Correlation Heatmap.

- Backend & API
  - FastAPI: The high-performance web framework used to wrap Python math engine. It receives the JSON payload from the frontend and routes it to the stress tester.

  - Uvicorn: The lightning-fast web server that actually hosts the FastAPI application and handles the concurrent network requests.

  - Pydantic: Used inside FastAPI to enforce strict data validation.

- Quantitative Math Engine
  - NumPy: Used for complex matrix multiplication, np.einsum vectorised simulations, and calculating medians/percentiles across thousands of Monte Carlo price paths instantly.

  - SciPy: Generate the Student-t distributions required for market crash simulations.

  - Pandas: Used to align time-series data, calculate daily log returns, build covariance matrices, and format the output data for the API.

- Data Ingestion
  - Alpaca API: The primary, premium data provider used to fetch perfectly accurate, timezone-aware daily OHLCV bars.

  - Yahoo Finance: The robust fallback mechanism. If Alpaca keys are missing, the pipeline gracefully defaults to scraping YFinance.

  - Requests: Used by the Streamlit frontend to send the sanitised data payload over the internal network to the FastAPI backend.

- DevOps & Infrastructure
  - Python 3.12: Version of Python programming language used for developement.

  - uv: Python package manager to resolve dependencies, lock them, and build the virtual environments.

  - Docker: Used to containerise the applications so they run identically on any operating system.

  - Docker Compose: The orchestrator that spins up both the UI container and the API container simultaneously and connects them via a secure internal Docker network.
