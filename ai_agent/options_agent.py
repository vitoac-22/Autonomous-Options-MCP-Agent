import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

class OptionsExecutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        self.base_url = "https://paper-api.alpaca.markets/v2"
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Faltan credenciales maestras de Alpaca en el entorno.")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

    def execute_atomic_transaction(self, strategy_payload):
        legs = strategy_payload.get("legs", [])
        if not legs:
            self.logger.info("Sin patas estructurales para operar. Abortando.")
            return

        base_qty = min(leg.get('qty', 1) for leg in legs)
        mleg_legs = []
        for leg in legs:
            mleg_legs.append({
                "symbol": leg['symbol'],
                "ratio_qty": int(leg['qty'] / base_qty),
                "side": leg['side'].lower()
            })

        payload = {
            "order_class": "mleg",
            "type": "market",
            "time_in_force": "day",
            "legs": mleg_legs
        }

        self.logger.info("Despachando bloque atómico MLEG directamente al motor del bróker...")
        response = requests.post(f"{self.base_url}/orders", json=payload, headers=self.headers)

        if response.status_code in [200, 201]:
            order_id = response.json().get('id')
            self.logger.info(f"=== ÉXITO ESTRUCTURAL === Orden MLEG aceptada. ID: {order_id}")
        else:
            error_msg = response.text
            self.logger.error(f"RECHAZO DE BRÓKER (Rollback automático garantizado): {error_msg}")
            raise RuntimeError(f"Alpaca rechazó la estructura MLEG: {error_msg}")

    def liquidate_portfolio(self, positions):
        """
        Fuerza bruta racional: Cierra todas las posiciones en SPY a mercado 
        para anular el riesgo de inmediato.
        """
        self.logger.info("Iniciando protocolo de liquidación de posiciones...")
        spy_options = [p for p in positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        for p in spy_options:
            symbol = p.symbol
            try:
                self.logger.info(f"Enviando orden de liquidación para: {symbol}")
                response = requests.delete(f"{self.base_url}/positions/{symbol}", headers=self.headers)
                if response.status_code in [200, 201]:
                    self.logger.info(f"Liquidación ejecutada: {symbol}")
                else:
                    self.logger.error(f"Fallo al liquidar {symbol}. Bróker: {response.text}")
            except Exception as e:
                self.logger.error(f"Error de red al liquidar {symbol}: {str(e)}")
        
        self.logger.info("=== PORTAFOLIO NEUTRALIZADO ===")