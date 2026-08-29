"""
The AI Execution Agent acting as the Portfolio Risk Manager.

Receives a pre-calculated MLEG payload and mathematical rationale from the quant
pipeline. Initializes a local MCP client, provisions the execution tool to the
Featherless LLM, and delegates the final dispatch decision to the AI.
"""
import os
import json
import asyncio
import logging
import requests
from dotenv import load_dotenv

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class OptionsExecutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.featherless_key = os.getenv('FEATHERLESS_API_KEY')
        if not self.featherless_key:
            raise ValueError("Missing FEATHERLESS_API_KEY in environment.")
            
        # Alpaca credentials strictly for the emergency liquidate fallback
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET') or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = "https://paper-api.alpaca.markets/v2"
        
        # Featherless is OpenAI-compatible. Route the base_url to their infrastructure.
        self.llm_client = AsyncOpenAI(
            api_key=self.featherless_key,
            base_url="https://api.featherless.ai/v1"
        )
        
        # Target a reliable tool-calling model on Featherless hub
        self.model_id = os.getenv('FEATHERLESS_MODEL', 'meta-llama/Meta-Llama-3.1-70B-Instruct')

    async def _evaluate_and_dispatch(self, payload: dict, rationale: str) -> str:
        """Spins up the MCP server, prompts the LLM, and handles the tool call."""
        self.logger.info("Initializing MCP Stdio Client and awakening Featherless LLM...")
        
        server_params = StdioServerParameters(
            command="python3",
            args=["mcp_server.py"],
            env=os.environ.copy()
        )

        try:
            # Compatibilidad con el SDK MCP local: Consumiendo el AsyncGenerator
            async for (read, write) in stdio_client(server_params):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    
                    target_tool_name = "dispatch_mleg_order"
                    
                    # Map MCP tool schema to OpenAI/Featherless tool format
                    llm_tools = []
                    for t in tools_response.tools:
                        if t.name == target_tool_name:
                            llm_tools.append({
                                "type": "function",
                                "function": {
                                    "name": t.name,
                                    "description": t.description,
                                    "parameters": t.inputSchema if hasattr(t, 'inputSchema') else {}
                                }
                            })

                    if not llm_tools:
                        raise RuntimeError(f"Tool '{target_tool_name}' not found on MCP server.")

                    system_prompt = f"""
                    You are the Chief Risk Officer of a quantitative hedge fund operating via MCP.
                    
                    The GARCH(1,1) engine has issued the following regime verdict and rationale:
                    {rationale}
                    
                    The convex optimizer has built the following atomic MLEG order payload:
                    {json.dumps(payload)}
                    
                    YOUR STRICT DIRECTIVE:
                    Execute the tool '{target_tool_name}' passing exactly the JSON payload (as a stringified object) to deploy the capital. 
                    Do not mutate strikes, ratios, or quantities. You act solely as the execution router.
                    """

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Authorize and execute the transaction via Alpaca now."}
                    ]
                    
                    self.logger.info("Requesting Featherless LLM audit and MLEG dispatch...")
                    
                    response = await self.llm_client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        tools=llm_tools,
                        temperature=0.0
                    )
                    
                    choice = response.choices[0]
                    
                    if choice.message.tool_calls:
                        for tool_call in choice.message.tool_calls:
                            self.logger.info(f"LLM authorized execution via MCP Tool: {tool_call.function.name}")
                            tool_args = json.loads(tool_call.function.arguments)
                            
                            result = await session.call_tool(
                                name=tool_call.function.name, 
                                arguments=tool_args
                            )
                            reply = "\n".join([c.text for c in result.content if c.type == "text"])
                            self.logger.info(f"Broker response via MCP: {reply}")
                            return reply
                            
                    self.logger.warning("LLM refused to execute the tool.")
                    return "Execution rejected by AI Agent."
                    
            return "Server connection terminated unexpectedly."
                    
        except Exception as e:
            self.logger.error(f"Critical failure in MCP-LLM bridge: {str(e)}")
            raise

    def submit_mleg_payload(self, payload: dict, rationale: str = "Approved by Risk Gates"):
        """Synchronous entry point called by the main pipeline."""
        return asyncio.run(self._evaluate_and_dispatch(payload, rationale))

    def liquidate_portfolio(self, positions):
        """
        Rational brute force: Close all SPY positions at market to neutralize risk.
        Bypasses AI to ensure immediate risk amputation.
        """
        self.logger.info("Initiating emergency liquidation protocol (Bypassing AI)...")
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        spy_options = [p for p in positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        for p in spy_options:
            symbol = p.symbol
            try:
                self.logger.info(f"Sending liquidation order for: {symbol}")
                response = requests.delete(f"{self.base_url}/positions/{symbol}", headers=headers)
                if response.status_code in [200, 201]:
                    self.logger.info(f"Liquidation executed: {symbol}")
                else:
                    self.logger.error(f"Failed to liquidate {symbol}. Broker: {response.text}")
            except Exception as e:
                self.logger.error(f"Network error liquidating {symbol}: {str(e)}")
        
        self.logger.info("=== PORTFOLIO NEUTRALIZED ===")