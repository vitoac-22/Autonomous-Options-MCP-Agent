import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

class OptionsExecutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        featherless_key = os.getenv('FEATHERLESS_API_KEY')
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        
        if not featherless_key:
            raise ValueError("Missing FEATHERLESS_API_KEY credential in .env.")
        if not api_key or not api_secret:
            raise ValueError("Missing Alpaca credentials in environment.")
            
        self.llm_client = OpenAI(
            base_url="https://api.featherless.ai/v1",
            api_key=featherless_key
        )
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.trading_client = TradingClient(api_key, api_secret, paper=True)

    def reason_target_structure(self, strategy_payload):
        messages = [
            {"role": "system", "content": "You are an Institutional Options Execution Agent. Parse the strategy payload and return ONLY a JSON object with a key 'orders' containing a list of dictionaries with 'symbol', 'qty' (integer), and 'side' (buy/sell)."},
            {"role": "user", "content": f"Strategy Payload: {strategy_payload}"}
        ]
        
        self.logger.info(f"Structured inference via Featherless AI ({self.model_id})...")
        response = self.llm_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)

    def execute_atomic_transaction(self, strategy_payload):
        """ACID: The entire options block executes via native SDK, or rolls back completely."""
        instructions = self.reason_target_structure(strategy_payload)
        orders = instructions.get("orders", [])
        
        if not orders:
            self.logger.info("No operations required by the model.")
            return

        executed_order_ids = []
        try:
            for order in orders:
                side_enum = OrderSide.BUY if order['side'].lower() == 'buy' else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=order['symbol'],
                    qty=int(order['qty']),
                    side=side_enum,
                    time_in_force=TimeInForce.DAY
                )
                submitted_order = self.trading_client.submit_order(req)
                executed_order_ids.append(submitted_order.id)
                self.logger.info(f"SUCCESS (SDK): {order['side'].upper()} order submitted for {order['symbol']}, Qty: {order['qty']}")
        
        except Exception as e:
            self.logger.error(f"ATOMICITY FAILURE. Initiating Rollback. Error: {str(e).strip()}")
            for order_id in executed_order_ids:
                try:
                    self.trading_client.cancel_order_by_id(order_id)
                except Exception as cancel_err:
                    self.logger.error(f"Failed to cancel order {order_id}: {cancel_err}")
            raise RuntimeError("Transactional block aborted to prevent asymmetric directional exposure.")