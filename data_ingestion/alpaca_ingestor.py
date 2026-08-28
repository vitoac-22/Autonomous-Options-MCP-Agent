import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

class AlpacaCredentials:
    """Base class for credential management (SRP)."""
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials missing in environment variables.")

class UnderlyingIngestor(AlpacaCredentials):
    """Handles spot market data for GARCH volatility forecasting (OCP)."""
    def __init__(self, ticker: str, start_date: str, end_date: str):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)

    def process_memory_data(self) -> pd.DataFrame:
        request_params = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=TimeFrame.Day,
            start=self.start_date,
            end=self.end_date
        )
        
        bars = self.client.get_stock_bars(request_params)
        df_ticker = bars.df.xs(self.ticker, level='symbol')
        
        if 'close' not in df_ticker.columns:
            raise KeyError(f"API failed to return 'close' column for {self.ticker}.")
            
        df_close = df_ticker[['close']].copy()
        df_close.rename(columns={'close': self.ticker}, inplace=True)
        
        # Atomicity: Return strict log returns or fail
        return np.log(df_close / df_close.shift(1)).dropna()

class OptionsChainIngestor(AlpacaCredentials):
    """Handles options chain retrieval for contract selection (OCP)."""
    def __init__(self, underlying_ticker: str):
        super().__init__()
        self.underlying_ticker = underlying_ticker
        self.client = OptionHistoricalDataClient(self.api_key, self.api_secret)

    def fetch_active_contracts(self) -> pd.DataFrame:
        request_params = OptionChainRequest(
            underlying_symbol=self.underlying_ticker
        )
        chain = self.client.get_option_chain(request_params)
        return chain.df