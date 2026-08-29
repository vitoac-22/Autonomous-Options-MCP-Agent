"""
The AI execution agent — Portfolio Risk Manager over MCP.

Receives an order payload the risk gates have already approved, together with
the mathematical rationale behind it, and asks the LLM to authorise dispatch
through a single MCP tool. The model never computes anything: the quant
pipeline builds the structure, the gates approve and size it, and the model's
only power is to release it or refuse.

Four fixes over the previous version.

`async for (read, write) in stdio_client(params)` raised
`TypeError: 'async for' requires an object with __aiter__ method` —
stdio_client is an @asynccontextmanager, so it needs `async with`. Every
approved trade died at dispatch. The existing test passed because its mock
supplied the `__aiter__` the real object lacks, so it confirmed the bug rather
than catching it.

The default model was a Llama ID. Featherless supports native tool calling only
on the Qwen 3 family and moonshotai/Kimi-K2-Instruct; Llama falls back to
simulated calling and never populates `tool_calls`, so the agent silently
declined every trade. The default now supports tool calling, and the model is
checked before dispatch.

A broker rejection was returned as a string, logged at INFO and discarded — the
pipeline then logged CYCLE COMPLETE and exited 0. Dispatch now parses the
result and raises on anything that is not a success.

The LLM authored the tool arguments and could alter the approved payload; the
prompt merely asked it not to. An approval hash now travels with the order and
the MCP tool refuses anything that does not match.
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

from ai_agent.execution_bridge import (
    canonical_payload_hash, is_tool_calling_model, verify_approval,
)

load_dotenv()

# Featherless serves native tool calling on Qwen 3 and Kimi-K2 only.
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DISPATCH_TOOL = "dispatch_mleg_order"


class OptionsExecutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Alpaca credentials, used by the liquidation fallback.
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = "https://paper-api.alpaca.markets/v2"

        self.model_id = os.getenv("FEATHERLESS_MODEL", DEFAULT_MODEL)

        # The LLM client is built on first use. Construction previously raised
        # without FEATHERLESS_API_KEY, which meant an emergency liquidation —
        # the one path that deliberately bypasses the AI — crashed when the AI
        # key was missing. The fire exit must not depend on the model.
        self._client = None

    def _llm_client(self):
        if self._client is None:
            key = os.getenv("FEATHERLESS_API_KEY")
            if not key:
                raise ValueError("Missing FEATHERLESS_API_KEY; cannot reach the LLM.")
            self._client = AsyncOpenAI(api_key=key,
                                       base_url="https://api.featherless.ai/v1")
        return self._client

    # ── dispatch ────────────────────────────────────────────────────────────
    async def _evaluate_and_dispatch(self, payload: dict, rationale: str,
                                     approval_hash: str) -> dict:
        if not is_tool_calling_model(self.model_id):
            raise RuntimeError(
                f"{self.model_id} does not support native tool calling on Featherless "
                "(only the Qwen 3 family and moonshotai/Kimi-K2-Instruct do). It would "
                "return no tool_calls and silently decline every trade. Set "
                "FEATHERLESS_MODEL to a supported model."
            )

        self.logger.info(f"Starting MCP server and LLM audit ({self.model_id})...")

        server_params = StdioServerParameters(
            command="python3", args=["mcp_server.py"], env=os.environ.copy(),
        )

        # stdio_client is an async CONTEXT MANAGER, not an async generator.
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": getattr(t, "inputSchema", {}) or {},
                        },
                    }
                    for t in (await session.list_tools()).tools
                    if t.name == DISPATCH_TOOL
                ]
                if not tools:
                    raise RuntimeError(f"Tool '{DISPATCH_TOOL}' not found on the MCP server.")

                system_prompt = f"""You are the Chief Risk Officer of a quantitative fund.

A GARCH(1,1) volatility engine produced this regime verdict:
{rationale}

Deterministic risk gates approved and sized this exact multi-leg order:
{json.dumps(payload)}

Call '{DISPATCH_TOOL}' with:
  payload_json  = the order above, serialised exactly as given
  approval_hash = {approval_hash}

You are the execution router. The structure has already passed every risk
check. Altering any strike, ratio, side or quantity will be rejected by the
dispatch tool, which verifies the payload against the approval hash."""

                response = await self._llm_client().chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Authorise and execute the transaction."},
                    ],
                    tools=tools,
                    temperature=0.0,
                )

                calls = response.choices[0].message.tool_calls
                if not calls:
                    raise RuntimeError(
                        "The LLM returned no tool call, so nothing was dispatched. "
                        f"Check that {self.model_id} supports native tool calling."
                    )

                for call in calls:
                    self.logger.info(f"LLM authorised dispatch via {call.function.name}")
                    args = json.loads(call.function.arguments)
                    args["approval_hash"] = approval_hash

                    # Verify here as well as in the MCP tool. The tool runs in a
                    # separate process and is the authoritative check, but
                    # catching tampering before dispatch keeps a mutated order
                    # from ever leaving this process.
                    verify_approval(args.get("payload_json", ""), approval_hash)

                    raw = await session.call_tool(name=call.function.name, arguments=args)
                    text = "\n".join(c.text for c in raw.content
                                     if getattr(c, "type", None) == "text")
                    self.logger.info(f"Broker response via MCP: {text}")

                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"Unparseable dispatch response: {text}")

                    # A rejection previously came back as a string the pipeline
                    # discarded before logging CYCLE COMPLETE and exiting 0.
                    if result.get("status") != "success":
                        raise RuntimeError(f"Dispatch failed: {text}")

                    return result

                raise RuntimeError("No dispatch result produced.")

    def submit_mleg_payload(self, payload: dict,
                            rationale: str = "Approved by risk gates") -> dict:
        """Synchronous entry point. Raises unless the broker accepted the order."""
        approval_hash = canonical_payload_hash(payload)
        return asyncio.run(self._evaluate_and_dispatch(payload, rationale, approval_hash))

    # ── emergency path (deliberately bypasses the LLM) ──────────────────────
    def liquidate_portfolio(self, positions):
        """Close all underlying option positions at market.

        Bypasses the AI entirely — the fire exit should not need the model's
        permission, or its API key.
        """
        self.logger.info("Emergency liquidation (AI bypassed)...")
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }
        legs = [p for p in positions if p.symbol.startswith("SPY") and len(p.symbol) > 5]

        failures = []
        for position in legs:
            symbol = position.symbol
            try:
                self.logger.info(f"Liquidating {symbol}")
                response = requests.delete(f"{self.base_url}/positions/{symbol}",
                                           headers=headers, timeout=30)
                if response.status_code in (200, 201):
                    self.logger.info(f"Liquidated {symbol}")
                else:
                    self.logger.error(f"Failed to liquidate {symbol}: {response.text}")
                    failures.append(symbol)
            except Exception as exc:
                self.logger.error(f"Network error liquidating {symbol}: {exc}")
                failures.append(symbol)

        if failures:
            self.logger.error(f"LIQUIDATION INCOMPLETE — still exposed: {failures}")
        else:
            self.logger.info("=== PORTFOLIO NEUTRALISED ===")
        return failures
