import os
import numpy as np
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

load_dotenv()

class AlpacaCredentials:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise ValueError("Credenciales maestras faltantes en .env")

class UnderlyingIngestor(AlpacaCredentials):
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

    def get_valid_occ_symbol(self, target_strike: float, option_type: str, target_dte: int = 30) -> str:
        req = GetOptionContractsRequest(
            underlying_symbols=[self.underlying_ticker],
            status="active"
        )
        contracts = self.trading_client.get_option_contracts(req)
        
        if not contracts.option_contracts:
            raise RuntimeError(f"BLOQUEO DE BRÓKER: Alpaca devolvió 0 contratos totales para {self.underlying_ticker}.")
            
        hoy = datetime.now().date()
        contratos_validos = []
        target_char = 'P' if option_type.lower() == 'put' else 'C'
        
        for c in contracts.option_contracts:
            symbol = str(c.symbol)
            
            # Validación por subyacente estricta
            if not symbol.startswith(self.underlying_ticker):
                continue
                
            # Identificación segura de Call/Put mediante atributos o lectura directa del símbolo OCC
            tipo_contrato = getattr(c, 'type', None) or getattr(c, 'option_type', None) or getattr(c, 'right', None)
            tipo_str = str(tipo_contrato).upper()
            
            match_tipo = False
            if option_type.lower() in tipo_str.lower():
                match_tipo = True
            elif target_char in symbol:
                # El formato OCC estándar incluye 'P' o 'C' antes del strike
                match_tipo = True
                
            if not match_tipo:
                continue
                
            # Extracción de fecha de expiración robusta
            exp_raw = getattr(c, 'expiration_date', None)
            exp_date = None
            if isinstance(exp_raw, date):
                exp_date = exp_raw
            elif exp_raw:
                try:
                    exp_date = datetime.strptime(str(exp_raw)[:10], '%Y-%m-%d').date()
                except Exception:
                    pass
            
            # Si el SDK no trae la fecha limpia, intentamos parsearla de los dígitos del símbolo OCC (YYMMDD)
            if not exp_date and len(symbol) >= len(self.underlying_ticker) + 6:
                try:
                    date_str = symbol[len(self.underlying_ticker):len(self.underlying_ticker)+6]
                    exp_date = datetime.strptime(date_str, '%y%m%d').date()
                except Exception:
                    continue

            if not exp_date:
                continue
                
            dte = (exp_date - hoy).days
            
            # Extracción del Strike
            strike = getattr(c, 'strike_price', None)
            if strike is not None:
                try:
                    strike = float(strike)
                except:
                    strike = None
            
            if strike is None and len(symbol) > 8:
                try:
                    # Últimos 8 dígitos del símbolo OCC representan el strike multiplicado por 1000
                    strike = float(symbol[-8:]) / 1000.0
                except Exception:
                    continue

            if dte >= 0 and strike is not None:
                contratos_validos.append({
                    'symbol': symbol,
                    'strike': strike,
                    'dte': dte
                })
                
        if not contratos_validos:
            raise RuntimeError(f"Fallo de liquidez: Cero contratos {option_type} válidos tras el barrido resiliente de memoria.")
            
        df = pd.DataFrame(contratos_validos)
        
        # Selección de la expiración más cercana disponible en el simulador
        min_dte = df['dte'].min()
        df_optimos_dte = df[df['dte'] == min_dte].copy()
        
        # Selección del strike más cercano al objetivo proyectado por GARCH
        df_optimos_dte['distancia_strike'] = abs(df_optimos_dte['strike'] - target_strike)
        contrato_optimo = df_optimos_dte.loc[df_optimos_dte['distancia_strike'].idxmin()]
        
        return contrato_optimo['symbol']