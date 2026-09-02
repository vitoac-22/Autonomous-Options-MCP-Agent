import numpy as np

class VolatilityCartographer:
    def __init__(self, current_price: float, garch_vol_daily: float, target_dte: int = 30):
        self.S = current_price
        self.vol = garch_vol_daily
        self.dte = target_dte
        
        # Proyección del movimiento esperado asumiendo caminata aleatoria (Random Walk)
        self.expected_move_pct = self.vol * np.sqrt(self.dte)
        self.expected_move_usd = self.S * self.expected_move_pct

    def map_iron_condor_strikes(self, wing_mult: float = 1.5):
        """
        Estructura defensiva con integración estricta de signos institucionales:
        - Long Put (Compra protección): wing_mult DE abajo (sign: 1)
        - Short Put (Vende riesgo): 1 DE abajo (sign: -1)
        - Short Call (Vende riesgo): 1 DE arriba (sign: -1)
        - Long Call (Compra protección): wing_mult DE arriba (sign: 1)

        wing_mult < 1.5 tightens the wings: the cheap far-OTM contracts
        sometimes carry no Greeks on the indicative feed, and the pipeline
        retries with a closer wing rather than inventing a delta.
        """
        long_put = round(self.S - (wing_mult * self.expected_move_usd), 0)
        short_put = round(self.S - self.expected_move_usd, 0)
        short_call = round(self.S + self.expected_move_usd, 0)
        long_call = round(self.S + (wing_mult * self.expected_move_usd), 0)

        return [
            {"type": "put", "side": "buy", "sign": 1, "target_strike": long_put},
            {"type": "put", "side": "sell", "sign": -1, "target_strike": short_put},
            {"type": "call", "side": "sell", "sign": -1, "target_strike": short_call},
            {"type": "call", "side": "buy", "sign": 1, "target_strike": long_call}
        ]

    def map_straddle_strikes(self):
        """
        Estructura ofensiva At-The-Money (ATM) para captura de expansión de varianza:
        - Long Call (sign: 1)
        - Long Put (sign: 1)
        """
        atm_strike = round(self.S, 0)
        return [
            {"type": "call", "side": "buy", "sign": 1, "target_strike": atm_strike},
            {"type": "put", "side": "buy", "sign": 1, "target_strike": atm_strike}
        ]