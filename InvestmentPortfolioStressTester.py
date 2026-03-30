import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

class PortfolioStressTester:
    def __init__(self, tickers, weights, base):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.base = base
        self.data = None
        self.logReturns = None

    def fetchData(self, startDate='2022-01-01', endDate="2026-01-01"):
        print(f"Fetching data for {self.tickers} from Yahoo Finance")
        # Data includes open price, close price, low price, high price, and volume information
        self.data = yf.download(self.tickers, start=startDate, end=endDate)
        closePrice = self.data['Close']
        
        if isinstance(closePrice, pd.Series):
            ticker_name = closePrice.name if closePrice.name and closePrice.name != 'Close' else self.tickers[0]
            closePrice = closePrice.to_frame(name=ticker_name)
            
        # Force column order to match self.tickers. yfinance returns columns alphabetically, 
        # which silently misaligns with self.weights and breaks risk math.
        valid_tickers = [t for t in self.tickers if t in closePrice.columns]
        closePrice = closePrice[valid_tickers]
        self.closePrices = closePrice
        # Compute daily log returns eliminate multi-period volatility, due to their compounding and time-additive properties
        self.logReturns = np.log(closePrice/closePrice.shift(1)).dropna()
        return self.logReturns
    
    def loadData(self, clean_close_prices):
        # The DataPipeline has already extracted 'Close' and validated the tickers
        self.closePrices = clean_close_prices
        # Compute daily log returns directly
        self.logReturns = np.log(self.closePrices/self.closePrices.shift(1)).dropna()
        return self.logReturns

    def runSimulation(self, dayHorizon=30, simulations=1500, shockVolatility=1.0, marketGap=0.0, meanShock=0.0, rebalance=False):
        # Historical parameters
        mu = self.logReturns.mean().values
        cov = self.logReturns.cov().values
        
        variance_expansion = np.diag(cov) * ((shockVolatility**2) - 1.0)
        ito_correction = 0.5 * variance_expansion
        daily_mean_shock = (meanShock / 252) - ito_correction 
        
        # Cholesky Decomposition & Student-t (Fat Tails)
        # We scale the Cholesky matrix by the shockVolatility
        L = np.linalg.cholesky(cov) * shockVolatility
        
        # df=3 represents the "fat tails" of market crashes
        df_t = 3 
        scale_factor = np.sqrt((df_t - 2) / df_t)
        # Generate random variables shape: (simulations, dayHorizon, num_assets)
        random_dist = t.rvs(df_t, size=(simulations, dayHorizon, len(self.tickers)))*scale_factor
        
        # Vectorized correlation mapping: bypass Python for-loop using einsum
        daily_asset_returns = np.einsum('ij, sdk -> sdi', L, random_dist) + mu + daily_mean_shock
        
        # Convert log returns back to multipliers
        daily_multipliers = np.exp(daily_asset_returns)
        
        if rebalance:
            portfolio_daily_multipliers = np.dot(daily_multipliers, self.weights)
            initial_gap = np.full((simulations, 1), 1 + marketGap)
            full_multipliers = np.concatenate([initial_gap, portfolio_daily_multipliers], axis=1)
            portfolioSimulation = (self.base * np.cumprod(full_multipliers, axis=1)).T
        else:
            initial_gap = np.ones((simulations, 1, len(self.tickers))) * (1 + marketGap)
            path_multipliers = np.concatenate([initial_gap, daily_multipliers], axis=1)
            asset_paths = np.cumprod(path_multipliers, axis=1)
            initial_dollars_per_asset = self.base * self.weights
            portfolioSimulation = np.sum(initial_dollars_per_asset * asset_paths, axis=2).T

        # Stats for metrics - preserved exactly so your dashboard doesn't break
        final_values = portfolioSimulation[-1, :]
        returns = (final_values / self.base) - 1
        realized_ann_return = np.mean(returns) * (252 / dayHorizon)
        realized_ann_vol = np.std(returns) * np.sqrt(252 / dayHorizon)
        
        return portfolioSimulation, realized_ann_return, realized_ann_vol

    def calculateRiskContribution(self):
        cov_matrix = self.logReturns.cov() * 252
        portfolio_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        marginal_contribution = np.dot(cov_matrix, self.weights) / portfolio_vol
        component_contribution = self.weights * marginal_contribution
        percent_contribution = component_contribution / portfolio_vol
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.sum(individual_vols * self.weights)
        diversification_ratio = weighted_vol / portfolio_vol
        return pd.Series(percent_contribution, index=self.tickers), diversification_ratio
    
    def plotResults(self, generalSimulation, extremeSimulation):
        fig, ax = plt.subplots(figsize=(12, 8))
        days = np.arange(generalSimulation.shape[0])
        # Compute percentiles for general simulation
        generalLower = np.percentile(generalSimulation, 5, axis=1)
        generalMedian = np.median(generalSimulation, axis=1)
        generalUpper = np.percentile(generalSimulation, 95, axis=1)
        # Compute percentiles for extreme simulation
        extremeLower = np.percentile(extremeSimulation, 5, axis=1)
        extremeMedian = np.median(extremeSimulation, axis=1)
        extremeUpper = np.percentile(extremeSimulation, 95, axis=1)
        # Plotting
        ax.fill_between(days, generalLower, generalUpper, color='blue', alpha=0.3, label=f"General 90% confidence")
        ax.plot(days, generalMedian, color='blue', lw=2, label='General median')
        ax.fill_between(days, extremeLower, extremeUpper, color='red', alpha=0.3, label='Extreme case')
        ax.plot(days, extremeMedian, color='red', lw=2, linestyle='--', label='Extreme median')
        ax.axhline(self.base, color='black', lw=1, linestyle='-')
        ax.set_title("Portfolio stress test: normal market vs. market crash")
        ax.set_xlabel("Days in future")
        ax.set_ylabel("Portfolio value (USD$)")
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Configuration
    tickers = ['GOOG', 'NVDA', 'AVGO', 'BTC-USD']
    weights = [0.05, 0.65, 0.25, 0.05]
    base = 23270

    tester = PortfolioStressTester(tickers, weights, base)
    tester.fetchData()
    # Normal market
    general, portfolioReturn, portfolioVolatility = tester.runSimulation(dayHorizon=30, simulations=1500, shockVolatility=1.0, marketGap=0.0, meanShock=0.0, rebalance=False)
    # Market crash
    crash, portfolioReturnCrash, portfolioVolatilityCrash = tester.runSimulation(dayHorizon=30, simulations=1500, shockVolatility=3.0, marketGap=-0.15, meanShock=-0.05, rebalance=False)
    # Risk analysis
    var_95 = base-np.percentile(general[-1, :], 5)
    stress_var = base-np.percentile(crash[-1, :], 5)
    
    #Visualise
    tester.plotResults(general, crash)