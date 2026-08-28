import numpy as np

class VolatilityCartographer:
    def __init__(self, current_price: float, garch_vol_daily: float, target_dte: int = 30):
        self.S = current_price
        self.vol = garch_vol_daily
        self.dte = target_dte
        
        # Proyección del movimiento esperado asumiendo caminata aleatoria (Random Walk)
        # Volatilidad en el periodo = Volatilidad Diaria * Raíz(Días)
        self.expected_move_pct = self.vol * np.sqrt(self.dte)
        self.expected_move_usd = self.S * self.expected_move_pct

    def map_iron_condor_strikes(self):
        """
        Estructura defensiva: 
        Vende riesgo a 1 Desviación Estándar (captura prima).
        Compra protección a 1.5 Desviaciones Estándar (limita pérdida máxima).
        """
        short_put = self.S - self.expected_move_usd
        long_put = self.S - (1.5 * self.expected_move_usd)
        
        short_call = self.S + self.expected_move_usd
        long_call = self.S + (1.5 * self.expected_move_usd)

        # Redondeamos al entero más cercano, ya que los strikes de SPY suelen ir de 1 en 1
        return {
            "long_put": round(long_put, 0),
            "short_put": round(short_put, 0),
            "short_call": round(short_call, 0),
            "long_call": round(long_call, 0)
        }

    def map_straddle_strikes(self):
        """
        Estructura ofensiva:
        Captura expansión de varianza operando At-The-Money (ATM).
        """
        atm_strike = round(self.S, 0)
        return {
            "long_call": atm_strike,
            "long_put": atm_strike
        }