import numpy as np
import pandas as pd
from arch import arch_model

class GarchVolatilityEngine:
    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df
        self.mle_params = {}

    def fit_model(self):
        scaled_returns = self.returns.iloc[:, 0] * 100 
        am = arch_model(scaled_returns, mean='Zero', vol='Garch', p=1, q=1, dist='skewstudent')
        fitted_model = am.fit(update_freq=5, disp='off')
        
        conditional_variance = (fitted_model.conditional_volatility ** 2) / 10000
        
        self.mle_params = {
            'alpha': fitted_model.params.get('alpha[1]', 0), 
            'beta': fitted_model.params.get('beta[1]', 0), 
            'omega': fitted_model.params.get('omega', 0),
            'persistence': fitted_model.params.get('alpha[1]', 0) + fitted_model.params.get('beta[1]', 0),
            'nu_kurtosis': fitted_model.params.get('eta', 0),
            'lambda_skewness': fitted_model.params.get('lambda', 0)
        }
        
        processed_df = pd.DataFrame({
            'log_return': self.returns.iloc[:, 0],
            'garch_variance': conditional_variance
        }, index=self.returns.index)
        
        return processed_df, self.mle_params, fitted_model

    def calculate_tail_risk(self, fitted_model, alpha=0.01):
        projected_vol = np.sqrt(fitted_model.forecast(horizon=1).variance.iloc[-1, 0] / 10000)
        std_residuals = fitted_model.resid / fitted_model.conditional_volatility
        extreme_quantile = np.nanquantile(std_residuals, alpha)
        dynamic_var = projected_vol * extreme_quantile
        
        return dynamic_var, projected_vol