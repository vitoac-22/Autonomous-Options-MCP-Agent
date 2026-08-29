from datetime import datetime, date
import logging

class PortfolioRiskManager:
    def __init__(self, positions):
        self.positions = positions
        self.logger = logging.getLogger(__name__)

    def evaluate_exit_conditions(self):
        """
        Retorna (bool, str): (Debe_Liquidarse, Razón)
        """
        spy_options = [p for p in self.positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        if not spy_options:
            return False, "Sin exposición en derivados."

        unrealized_pl = sum(float(p.unrealized_pl) for p in spy_options)
        hoy = datetime.now().date()

        # 1. Riesgo Gamma (Tiempo al Vencimiento)
        for p in spy_options:
            symbol = p.symbol
            try:
                date_str = symbol[3:9]
                exp_date = datetime.strptime(date_str, '%y%m%d').date()
                dte = (exp_date - hoy).days
                if dte <= 5:
                    return True, f"Riesgo Gamma inminente (DTE={dte}). Liquidación para neutralizar aceleración direccional."
            except Exception as e:
                self.logger.warning(f"Error parseando DTE para {symbol}: {e}")

        # 2. Take Profit Táctico
        if unrealized_pl > 150.0:  # Umbral dinámico ajustable según tu apetito
            return True, f"Take Profit alcanzado (PnL: +${unrealized_pl:.2f}). Captura de volatilidad exitosa."

        # 3. Stop Loss Estructural
        if unrealized_pl < -350.0: # Corte de sangría antes del VaR extremo
            return True, f"Stop Loss quebrado (PnL: ${unrealized_pl:.2f}). Amputación de riesgo ejecutada."

        return False, f"Portafolio estable. PnL actual: ${unrealized_pl:.2f}. Manteniendo estructura."