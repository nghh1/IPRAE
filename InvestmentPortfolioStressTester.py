import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

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
        # Compute daily log returns eliminate multi-period volatility, due to their compounding and time-additive properties
        self.logReturns = np.log(closePrice/closePrice.shift(1)).dropna()
        return self.logReturns
    
    def runMonteCarloSimulation(self, dayHorizon=30, simulations=1500, shockVolatility=1.0, marketGap=0.0, meanShock=0.0):
        """
        dayHorizon: days of simulation, default 30 days
        simulations: number of simulations to run, default 1500 times
        shockVolatility: default 1.0, edge cases >1.0 increases volatility
        marketGap: default 0.0, negative for an immediate n% drop
        meanShock: default 0.0, shifts the daily expected return if market crash occurs
        """
        # Compute mean vector and covariance matrix
        mean_logReturns = self.logReturns.mean()+meanShock
        covariance_logReturns = self.logReturns.cov()*(shockVolatility**2)
        # 252 market trading days in a year 
        portfolioReturn = np.sum(mean_logReturns*self.weights)*252
        portfolioVolatility = np.sqrt(np.dot(np.array(self.weights).T, np.dot(covariance_logReturns*252, self.weights)))

        startPrice = self.base*(1+marketGap)
        portfolioSimulation = np.zeros((dayHorizon, simulations))
        for i in range(simulations):
            # Generate random daily returns using Multivariate Gaussian Distribution
            shocks = np.random.multivariate_normal(mean_logReturns, covariance_logReturns, dayHorizon)
            # Calculate daily portfolio growth while assuming weights unchanged
            daily_portfolio_returns = np.dot(shocks, self.weights)
            # Cumulative growth over the n-day path
            portfolioSimulation[:, i] = startPrice*np.exp(np.cumsum(daily_portfolio_returns))
        return portfolioSimulation, portfolioReturn, portfolioVolatility
    
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
    general, portfolioReturn, portfolioVolatility = tester.runMonteCarloSimulation(dayHorizon=30, simulations=1500)
    # Market crash
    crash, portfolioReturnCrash, portfolioVolatilityCrash = tester.runMonteCarloSimulation(dayHorizon=30, simulations=1500, shockVolatility=3.0, marketGap=-0.15, meanShock=-0.05)
    # Risk analysis
    var_95 = base-np.percentile(general[-1, :], 5)
    stress_var = base-np.percentile(crash[-1, :], 5)
    
    #Visualise
    tester.plotResults(general, crash)
