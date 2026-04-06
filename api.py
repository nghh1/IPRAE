import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session
from InvestmentPortfolioStressTester import PortfolioStressTester
from DataPipeline import MarketDataPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Risk Analyser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501", "*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./iprae_community.db")
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

class DBPortfolio(Base):
    __tablename__ = "portfolio_updated"
    id = Column(Integer, primary_key=True, index=True)
    author = Column(String, index=True)
    name = Column(String)
    tickers = Column(JSON)
    weights = Column(JSON)
    normal_var = Column(Float)
    stress_var = Column(Float)
    sortino_ratio = Column(Float)
    base_capital = Column(Float)
    start_date = Column(String)
    end_date = Column(String)
    day_horizon = Column(Integer)
    simulations = Column(Integer)
    shock_volatility = Column(Float)
    market_gap = Column(Float)
    mean_shock = Column(Float)
    rebalance = Column(Boolean)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PortfolioPublish(BaseModel):
    author: str
    name: str
    tickers: List[str]
    weights: List[float]
    normal_var: float
    stress_var: float
    sortino_ratio: float
    base_capital: float
    start_date: str
    end_date: str
    day_horizon: int
    simulations: int
    shock_volatility: float
    market_gap: float
    mean_shock: float
    rebalance: bool

@app.post("/api/v1/community/publish")
async def publish_portfolio(portfolio: PortfolioPublish, db: Session = Depends(get_db)):
    try:
        db_portfolio = DBPortfolio(
            author=portfolio.author,
            name=portfolio.name,
            tickers=portfolio.tickers,
            weights=portfolio.weights,
            normal_var=portfolio.normal_var,
            stress_var=portfolio.stress_var,
            sortino_ratio=portfolio.sortino_ratio,
            base_capital=portfolio.base_capital,
            start_date=portfolio.start_date,
            end_date=portfolio.end_date,
            day_horizon=portfolio.day_horizon,
            simulations=portfolio.simulations,
            shock_volatility=portfolio.shock_volatility,
            market_gap=portfolio.market_gap,
            mean_shock=portfolio.mean_shock,
            rebalance=portfolio.rebalance
        )
        db.add(db_portfolio)
        db.commit()
        db.refresh(db_portfolio)
        return {"message": "Porfolio published successfully", "status": "success", "id": db_portfolio.id}
    except Exception as e:
        logger.error(f"Failed to publish portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail="Database insertion failed")

@app.get("/api/v1/community/top_portfolios")
async def get_community_portfolios(db: Session = Depends(get_db)):
    portfolios = db.query(DBPortfolio).order_by(DBPortfolio.sortino_ratio.desc()).limit(10).all()
    return portfolios

@app.delete("/api/v1/community/portfolio/{portfolio_id}")
async def delete_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(DBPortfolio).filter(DBPortfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    db.delete(portfolio)
    db.commit()
    return {"message": "Portfolio deleted successfully", "status": "success"}

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