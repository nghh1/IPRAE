import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class MarketDataPipeline:
    def __init__(self, tickers, start_date, end_date, api_key=None, api_secret=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.api_secret = api_secret

    def fetch_data(self):
        # Check if you have a premium API key.
        if self.api_key:
            print("Fetching data from Alpaca API")
            return self._fetch_premium_data()
        else:
            print("Fetching data from yfinance API")
            return self._fetch_yfinance_fallback()

    def _fetch_yfinance_fallback(self):
        # We use threads=True to speed up concurrent downloads
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, threads=True)
        close_prices = raw_data['Close']
        if isinstance(close_prices, pd.Series):
            ticker_name = close_prices.name if close_prices.name and close_prices.name != 'Close' else self.tickers[0]
            close_prices = close_prices.to_frame(name=ticker_name)
            
        return self._clean_and_validate(close_prices)

    def _fetch_premium_data(self):
        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        request_params = StockBarsRequest(
            symbol_or_symbols=self.tickers,
            timeframe=TimeFrame.Day,
            start=pd.to_datetime(self.start_date),
            end=pd.to_datetime(self.end_date)
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df
        close_prices = df.reset_index().pivot(index='timestamp', columns='symbol', values='close')
        close_prices.index = close_prices.index.tz_localize(None).normalize()
        return self._clean_and_validate(close_prices)

    def _clean_and_validate(self, df):
        # Sanitise the data before the math engine sees it.
        # Force column order to match requested tickers
        valid_tickers = [t for t in self.tickers if t in df.columns]
        df = df[valid_tickers]
        # Handle missing days
        df = df.ffill()
        # Drop assets that don't have enough historical data
        # If an asset is missing more than 10% of its history, we must flag it.
        threshold = len(df) * 0.90
        df = df.dropna(axis=1, thresh=threshold)
        # Drop any remaining NaNs at the very beginning of the dataset
        df = df.dropna()
        # Sanity Check: Ensure no negative prices or zero prices (which break log math)
        if (df <= 0).any().any():
            raise ValueError("Data corruption detected: Asset prices cannot be zero or negative.")
        return df
