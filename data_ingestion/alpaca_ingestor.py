import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from data_ingestion.market_data_window import safe_end_timestamp

load_dotenv()

class AlpacaCredentials:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise ValueError("Credenciales maestras faltantes en .env")

class UnderlyingIngestor(AlpacaCredentials):
    def __init__(self, ticker: str, start_date: str, end_date=None):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        # Alpaca's free plan refuses bars inside the last 15 minutes
        # ("subscription does not permit querying recent SIP data"), so the
        # window always stops short of now regardless of what the caller asks
        # for. Passing today's date failed outright during market hours.
        self.end_date = safe_end_timestamp()
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
        df_close = df_ticker[['close']].copy()
        df_close.rename(columns={'close': self.ticker}, inplace=True)
        return np.log(df_close / df_close.shift(1)).dropna()

    def get_latest_price(self) -> float:
        request_params = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=TimeFrame.Day,
            start=self.start_date,
            end=self.end_date
        )
        bars = self.client.get_stock_bars(request_params)
        return float(bars.df.xs(self.ticker, level='symbol').iloc[-1]['close'])

class OptionsContractResolver(AlpacaCredentials):
    def __init__(self, underlying_ticker: str):
        super().__init__()
        self.underlying_ticker = underlying_ticker.upper()
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)

    def get_valid_occ_symbol(self, target_strike: float, option_type: str, exclude_symbols: set = None) -> str:
        exclude_symbols = exclude_symbols or set()
        hoy = datetime.now().date()
        
        # 1. Precisión Cuantitativa: Acotar la superficie de volatilidad en la petición REST
        min_exp = hoy + timedelta(days=3)
        target_char = 'P' if option_type.lower() == 'put' else 'C'
        
        # Ventana de liquidez ampliada a +/- $15 para capturar las alas del Iron Condor
        req = GetOptionContractsRequest(
            underlying_symbols=[self.underlying_ticker],
            status="active",
            expiration_date_gte=min_exp.strftime('%Y-%m-%d'),
            strike_price_gte=str(round(float(target_strike) - 15.0, 2)),
            strike_price_lte=str(round(float(target_strike) + 15.0, 2))
        )
        
        res = self.trading_client.get_option_contracts(req)
        contracts = res.option_contracts
            
        if not contracts:
            raise RuntimeError(f"Fallo de liquidez en el servidor: Cero contratos devueltos cerca del strike {target_strike} para {self.underlying_ticker}.")
            
        contratos_validos = []
        for c in contracts:
            symbol = str(c.symbol)
            if symbol in exclude_symbols:
                continue
                
            # 2. Filtrado determinista de Tipo (Call vs Put) a nivel de cadena OCC
            if target_char not in symbol:
                continue
                
            # 3. Extracción robusta de variables
            strike = getattr(c, 'strike_price', None)
            if strike is not None:
                try:
                    strike = float(strike)
                except:
                    strike = None
            if strike is None and len(symbol) >= 8:
                try:
                    strike = float(symbol[-8:]) / 1000.0
                except:
                    continue
            if strike is None:
                continue
                
            exp_raw = getattr(c, 'expiration_date', None)
            if not exp_raw:
                continue
                
            try:
                exp_date = exp_raw if isinstance(exp_raw, date) else datetime.strptime(str(exp_raw)[:10], '%Y-%m-%d').date()
                dte = (exp_date - hoy).days
            except:
                continue

            if dte < 3:
                continue

            contratos_validos.append({
                'symbol': symbol,
                'strike': strike,
                'dte': dte
            })
                
        if not contratos_validos:
            raise RuntimeError(f"Fallo de asignación tras el filtrado estricto para {option_type.upper()} a ${target_strike}.")
            
        df = pd.DataFrame(contratos_validos)
        
        # 4. Snapping Institucional: Minimizar deslizamiento buscando el strike exacto
        df['distancia'] = abs(df['strike'] - target_strike)
        df = df.sort_values(['distancia', 'dte'])
        
        for _, row in df.iterrows():
            if row['symbol'] not in exclude_symbols:
                return row['symbol']
                
        return df.iloc[0]['symbol']