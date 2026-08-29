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

    def submit_mleg_payload(self, payload: dict):
        """Send a prebuilt multi-leg order.

        The payload comes from ai_agent.mleg_payload, which is a pure function
        and unit-tested. Building it here inline was how the order ended up
        missing the top-level `qty` and per-leg `position_intent` that Alpaca
        requires for order_class="mleg" — the reason every attempt was rejected
        422 — and how leg ratios got truncated by integer division.

        Atomicity is the broker's: an mleg order fills as one unit or not at
        all, so there is no partial-fill window and no rollback to get wrong.
        """
        self.logger.info(f"Submitting atomic MLEG order: qty={payload.get('qty')} "
                         f"legs={len(payload.get('legs', []))} "
                         f"limit={payload.get('limit_price', 'market')}")
        response = requests.post(f"{self.base_url}/orders",
                                 json=payload, headers=self.headers, timeout=30)

        if response.status_code in (200, 201):
            order_id = response.json().get("id")
            self.logger.info(f"ACCEPTED by broker. Order ID: {order_id}")
            return order_id

        self.logger.error(f"BROKER REJECTED: {response.status_code} {response.text}")
        raise RuntimeError(f"Alpaca rejected the MLEG order: {response.text}")

    def execute_atomic_transaction(self, strategy_payload):
        """Deprecated. Build the payload with mleg_payload.build_mleg_payload and
        call submit_mleg_payload instead — this path could not construct a valid
        order."""
        raise NotImplementedError(
            "execute_atomic_transaction built an invalid mleg payload (no top-level "
            "qty, no position_intent, truncated ratios). Use "
            "ai_agent.mleg_payload.build_mleg_payload() then submit_mleg_payload()."
        )

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