import numpy as np
import scipy.stats as si

class BlackScholesPricer:
    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            return (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        if option_type == "put":
            return (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

class OptionsStrategySelector:
    def __init__(self, var_threshold: float = -0.035):
        self.var_threshold = var_threshold

    def determine_strategy(self, dynamic_var: float):
        if dynamic_var < self.var_threshold:
            return {
                "regime": "high_volatility",
                "strategy": "long_straddle",
                "legs": [{"type": "call", "side": "buy"}, {"type": "put", "side": "buy"}],
                "rationale": "VaR breach detected. Structuring straddle to capture absolute variance expansion."
            }
        return {
            "regime": "volatility_compression",
            "strategy": "iron_condor",
            "legs": [
                {"type": "put", "side": "buy", "moneyness": "OTM"},
                {"type": "put", "side": "sell", "moneyness": "OTM"},
                {"type": "call", "side": "sell", "moneyness": "OTM"},
                {"type": "call", "side": "buy", "moneyness": "OTM"}
            ],
            "rationale": "Stable regime detected. Structuring Iron Condor to capture volatility premium."
        }