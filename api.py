from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

from InvestmentPortfolioStressTester import PortfolioStressTester
from DataPipeline import MarketDataPipeline

app = FastAPI(title="Risk Analyser API", version="1.0.0")

class SimulationRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    base: float
    start_date: str
    end_date: str
    day_horizon: int
    simulations: int
    shock_volatility: float
    market_gap: float
    mean_shock: float
    rebalance: bool
    market_data: dict

@app.post("/api/v1/simulate")
def run_risk_simulation(req: SimulationRequest):
    try:
        clean_data = pd.DataFrame(req.market_data)
        clean_data.index = pd.to_datetime(clean_data.index)
        valid_tickers = list(clean_data.columns)
        if len(valid_tickers) != len(req.tickers):
            missing = list(set(req.tickers) - set(valid_tickers))
            raise ValueError(f"Invalid data for assets: {', '.join(missing)}. Please check for typos or recently listed IPOs.")
        
        # Initialise the Engine
        tester = PortfolioStressTester(req.tickers, req.weights, req.base)
        tester.loadData(clean_data)
        # Run the simulations
        np.random.seed(42)
        general, ann_ret, ann_vol = tester.runSimulation(
            req.day_horizon, req.simulations, rebalance=req.rebalance
        )
        np.random.seed(42)
        crash, ann_ret_crash, ann_vol_crash = tester.runSimulation(
            req.day_horizon, req.simulations, 
            shockVolatility=req.shock_volatility, 
            marketGap=req.market_gap, 
            meanShock=req.mean_shock, 
            rebalance=req.rebalance
        )
        # Get the Risk Contributions for the Dashboard
        risk_contrib, div_ratio = tester.calculateRiskContribution()
        # Package everything the frontend needs into JSON-friendly formats
        # We use .fillna(0) to prevent JSON errors with missing data
        return {
            "status": "success",
            "normal_simulation": general.tolist(),
            "stress_simulation": crash.tolist(),
            "historical_prices": clean_data.fillna(0).to_dict(),
            "correlation_matrix": tester.logReturns.corr().fillna(0).to_dict(),
            "risk_contribution": risk_contrib.to_dict(),
            "diversification_ratio": div_ratio
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))