import os
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import numpy as np
import logging

from InvestmentPortfolioStressTester import PortfolioStressTester
from DataPipeline import MarketDataPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

@app.get("/health")
def health_check():
    # Endpoint for Docker to verify the API is running
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/simulate")
async def run_risk_simulation(req: SimulationRequest):
    try:
        logger.info(f"Received simulation request for {len(req.tickers)} assets: {req.tickers}")
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        logger.info("Initialising data pipeline...")
        pipeline = MarketDataPipeline(req.tickers, req.start_date, req.end_date, api_key, api_secret)
        clean_data = await run_in_threadpool(pipeline.fetch_data)
        valid_tickers = list(clean_data.columns)
        if len(valid_tickers) != len(req.tickers):
            missing = list(set(req.tickers) - set(valid_tickers))
            logger.error(f"Invalid data detected for assets: {', '.join(missing)}")
            raise ValueError(f"Invalid data for assets: {', '.join(missing)}. Please check for typos or recently listed IPOs.")
        
        # Initialise the Engine
        tester = PortfolioStressTester(req.tickers, req.weights, req.base)
        tester.loadData(clean_data)
        # Run the simulations
        logger.info("Executing normal simulation engine...")
        np.random.seed(42)
        general, ann_ret, ann_vol = await run_in_threadpool(
            tester.runSimulation, req.day_horizon, req.simulations, rebalance=req.rebalance
        )
        logger.info("Executing stress simulation engine...")
        np.random.seed(42)
        crash, ann_ret_crash, ann_vol_crash = await run_in_threadpool(
            tester.runSimulation,
            req.day_horizon, 
            req.simulations, 
            shockVolatility=req.shock_volatility, 
            marketGap=req.market_gap, 
            meanShock=req.mean_shock, 
            rebalance=req.rebalance
        )
        # Get the Risk Contributions for the Dashboard
        risk_contrib, div_ratio = tester.calculateRiskContribution()
        # Package everything the frontend needs into JSON-friendly formats
        logger.info("Simulation complete, returning result payload.")
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
        logger.error(f"Simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))