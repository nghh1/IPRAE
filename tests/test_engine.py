import numpy as np
import pandas as pd
from InvestmentPortfolioStressTester import PortfolioStressTester

def test_weight_normalisation():
    # Test that the engine automatically forces weights to equal 1.0.
    # We pass weights that sum to 1.5
    tester = PortfolioStressTester(tickers=["AAPL", "GOOG"], weights=[1.0, 0.5], base=100)
    
    # The engine should have automatically normalized them to 0.666 and 0.333
    assert np.isclose(np.sum(tester.weights), 1.0)
    assert np.isclose(tester.weights[0], 0.66666667)

def test_simulation_output_shape():
    #Test that the engine outputs the correct array sizes.
    tester = PortfolioStressTester(tickers=["AAPL", "GOOG"], weights=[0.5, 0.5], base=100)
    # Create fake data for 10 days
    dates = pd.date_range("2023-01-01", periods=10)
    fake_data = pd.DataFrame({
        "AAPL": np.random.uniform(100, 150, 10),
        "GOOG": np.random.uniform(50, 80, 10)
    }, index=dates)
    tester.loadData(fake_data)
    # Run a small simulation (5 days, 10 paths)
    paths, ann_ret, ann_vol = tester.runSimulation(dayHorizon=5, simulations=10)
    # The output array should be (6, 10)
    assert paths.shape == (6, 10)
    assert isinstance(ann_ret, float)
    assert isinstance(ann_vol, float)