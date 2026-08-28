import os
import json
import logging
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class OptionsExecutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        featherless_key = os.getenv('FEATHERLESS_API_KEY')
        
        if not featherless_key:
            raise ValueError("Missing FEATHERLESS_API_KEY credential in .env.")
            
        self.llm_client = OpenAI(
            base_url="https://api.featherless.ai/v1",
            api_key=featherless_key
        )
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"

    def _run_cli(self, command):
        """Isolated native execution for Alpaca CLI compliance."""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

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
        """ACID: The entire options block executes, or rolls back completely."""
        instructions = self.reason_target_structure(strategy_payload)
        orders = instructions.get("orders", [])
        
        if not orders:
            self.logger.info("No operations required by the model.")
            return

        executed_symbols = []
        try:
            for order in orders:
                # Strict Hackathon Compliance: Alpaca CLI usage with positional symbol
                cmd = f"alpaca order submit {order['symbol']} -q {order['qty']} --side {order['side']} --type market --tif day"
                self._run_cli(cmd)
                executed_symbols.append(order['symbol'])
                self.logger.info(f"SUCCESS (CLI): {order['side'].upper()} order submitted for {order['symbol']}, Qty: {order['qty']}")
        
        except Exception as e:
            self.logger.error(f"ATOMICITY FAILURE. Initiating Rollback. Error: {str(e).strip()}")
            for symbol in executed_symbols:
                self._run_cli(f"alpaca order cancel {symbol}")
            raise RuntimeError("Transactional block aborted to prevent asymmetric directional exposure.")