"""
Tests for the LLM/MCP execution bridge.

Four defects on main, two of which independently stop any trade:

1. `async for (read, write) in stdio_client(params)` — stdio_client is an
   @asynccontextmanager, not an async generator. Raises
   TypeError: 'async for' requires an object with __aiter__ method.
   Every approved trade dies at dispatch.

2. The default model is `meta-llama/Meta-Llama-3.1-70B-Instruct`. Featherless
   supports native tool calling only on the Qwen 3 family and
   moonshotai/Kimi-K2-Instruct; Llama gets simulated calling via prompting and
   returns no `tool_calls`. So even with (1) fixed the LLM never invokes the
   tool and nothing trades.

3. A broker rejection is returned as a string, logged at INFO, and discarded —
   the pipeline then logs CYCLE COMPLETE and exits 0. Same failure-reported-as-
   success shape as the bug that made CI show green on Friday's failed run.

4. The LLM authors the tool arguments, so it can alter the payload the risk
   gates approved. The system prompt asks it not to; nothing enforces it.

The mock below is a REAL @asynccontextmanager. The existing suite passes an
`AsyncGeneratorMock` that supplies the `__aiter__` the real object lacks, so it
confirms defect 1 instead of catching it.
"""
import json
import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch, MagicMock

from ai_agent.execution_bridge import (
    canonical_payload_hash, verify_approval, ApprovalMismatch,
    TOOL_CALLING_MODELS, is_tool_calling_model,
)

PAYLOAD = {
    "order_class": "mleg",
    "qty": "2",
    "type": "limit",
    "limit_price": "0.95",
    "time_in_force": "day",
    "legs": [
        {"symbol": "SPY260904P00751000", "ratio_qty": "1", "side": "buy",
         "position_intent": "buy_to_open"},
        {"symbol": "SPY260904C00782000", "ratio_qty": "1", "side": "sell",
         "position_intent": "sell_to_open"},
    ],
}


class TestApprovalHash(unittest.TestCase):
    def test_hash_is_stable_across_calls(self):
        self.assertEqual(canonical_payload_hash(PAYLOAD),
                         canonical_payload_hash(PAYLOAD))

    def test_hash_ignores_key_ordering(self):
        """The LLM may re-serialise the JSON; key order must not matter."""
        reordered = json.loads(json.dumps(PAYLOAD))
        reordered = {k: reordered[k] for k in reversed(list(reordered))}
        self.assertEqual(canonical_payload_hash(PAYLOAD),
                         canonical_payload_hash(reordered))

    def test_hash_changes_when_quantity_is_altered(self):
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["qty"] = "99"
        self.assertNotEqual(canonical_payload_hash(PAYLOAD),
                            canonical_payload_hash(tampered))

    def test_hash_changes_when_a_strike_is_altered(self):
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["legs"][0]["symbol"] = "SPY260904P00700000"
        self.assertNotEqual(canonical_payload_hash(PAYLOAD),
                            canonical_payload_hash(tampered))

    def test_hash_changes_when_a_side_is_flipped(self):
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["legs"][0]["side"] = "sell"
        self.assertNotEqual(canonical_payload_hash(PAYLOAD),
                            canonical_payload_hash(tampered))


class TestVerifyApproval(unittest.TestCase):
    def test_untouched_payload_passes(self):
        verify_approval(json.dumps(PAYLOAD), canonical_payload_hash(PAYLOAD))

    def test_tampered_payload_is_refused(self):
        """The gates approved one structure; only that one may be dispatched."""
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["qty"] = "500"
        with self.assertRaises(ApprovalMismatch):
            verify_approval(json.dumps(tampered), canonical_payload_hash(PAYLOAD))

    def test_missing_hash_is_refused(self):
        with self.assertRaises(ApprovalMismatch):
            verify_approval(json.dumps(PAYLOAD), "")

    def test_malformed_json_is_refused(self):
        with self.assertRaises(ApprovalMismatch):
            verify_approval("{not json", canonical_payload_hash(PAYLOAD))


class TestModelSelection(unittest.TestCase):
    def test_llama_is_not_a_tool_calling_model(self):
        """Featherless: Llama gets simulated calling only, returns no tool_calls."""
        self.assertFalse(is_tool_calling_model("meta-llama/Meta-Llama-3.1-70B-Instruct"))

    def test_qwen3_and_kimi_are_tool_calling_models(self):
        self.assertTrue(is_tool_calling_model("Qwen/Qwen3-32B"))
        self.assertTrue(is_tool_calling_model("moonshotai/Kimi-K2-Instruct"))

    def test_the_default_model_supports_tool_calling(self):
        from ai_agent.options_agent import DEFAULT_MODEL
        self.assertTrue(is_tool_calling_model(DEFAULT_MODEL),
                        msg=f"{DEFAULT_MODEL} cannot return tool_calls on Featherless")

    def test_known_good_list_is_not_empty(self):
        self.assertTrue(TOOL_CALLING_MODELS)


class TestAgentConstruction(unittest.TestCase):
    @patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}, clear=True)
    def test_liquidation_does_not_require_the_llm_key(self):
        """The fire exit must not depend on the AI. Construction previously
        raised without FEATHERLESS_API_KEY, so an emergency flatten crashed."""
        from ai_agent.options_agent import OptionsExecutionAgent
        agent = OptionsExecutionAgent()          # must not raise
        self.assertIsNotNone(agent)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}, clear=True)
    def test_dispatch_still_requires_the_llm_key(self):
        from ai_agent.options_agent import OptionsExecutionAgent
        # clear=True is not sufficient on its own: options_agent calls
        # load_dotenv() at import, which repopulates os.environ from .env inside
        # the patched context. Remove the key explicitly so this asserts the
        # code path rather than the developer's local .env.
        os.environ.pop("FEATHERLESS_API_KEY", None)
        agent = OptionsExecutionAgent()
        with self.assertRaises(ValueError):
            agent._llm_client()


class TestAsyncContextManagerUsage(unittest.TestCase):
    """The mock here is a REAL @asynccontextmanager — no __aiter__.

    Code using `async for` fails these tests, which is what should have
    happened on main.
    """

    def _run(self, tool_calls, tool_result_text):
        import asyncio
        from ai_agent.options_agent import OptionsExecutionAgent

        @asynccontextmanager
        async def fake_stdio(_params):
            yield ("read", "write")

        class FakeSession:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def initialize(self): return None
            async def list_tools(self):
                tool = MagicMock()
                tool.name = "dispatch_mleg_order"
                tool.description = "dispatch"
                tool.inputSchema = {"type": "object", "properties": {}}
                return MagicMock(tools=[tool])
            async def call_tool(self, name, arguments):
                return MagicMock(content=[MagicMock(type="text", text=tool_result_text)])

        class FakeCompletions:
            async def create(self, **kw):
                msg = MagicMock(tool_calls=tool_calls)
                return MagicMock(choices=[MagicMock(message=msg)])

        class FakeLLM:
            chat = MagicMock(completions=FakeCompletions())

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s",
                                       "FEATHERLESS_API_KEY": "f"}, clear=True), \
             patch("ai_agent.options_agent.stdio_client", fake_stdio), \
             patch("ai_agent.options_agent.ClientSession", FakeSession):
            agent = OptionsExecutionAgent()
            agent._client = FakeLLM()
            return asyncio.run(agent._evaluate_and_dispatch(
                PAYLOAD, "test rationale", canonical_payload_hash(PAYLOAD)))

    def _tool_call(self, args: dict):
        call = MagicMock()
        call.function.name = "dispatch_mleg_order"
        call.function.arguments = json.dumps(args)
        return call

    def test_successful_dispatch_returns_the_order_id(self):
        args = {"payload_json": json.dumps(PAYLOAD),
                "approval_hash": canonical_payload_hash(PAYLOAD)}
        result = self._run([self._tool_call(args)],
                           json.dumps({"status": "success", "order_id": "abc123"}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["order_id"], "abc123")

    def test_broker_rejection_raises_instead_of_reporting_success(self):
        """A rejection previously returned a string that the pipeline discarded
        before logging CYCLE COMPLETE and exiting 0."""
        args = {"payload_json": json.dumps(PAYLOAD),
                "approval_hash": canonical_payload_hash(PAYLOAD)}
        with self.assertRaises(RuntimeError):
            self._run([self._tool_call(args)],
                      json.dumps({"status": "rejected", "error": "422 bad payload"}))

    def test_llm_declining_to_call_the_tool_raises(self):
        with self.assertRaises(RuntimeError):
            self._run(None, "")

    def test_llm_tampering_with_the_payload_is_refused(self):
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["qty"] = "999"
        args = {"payload_json": json.dumps(tampered),
                "approval_hash": canonical_payload_hash(PAYLOAD)}
        with self.assertRaises((ApprovalMismatch, RuntimeError)):
            self._run([self._tool_call(args)],
                      json.dumps({"status": "success", "order_id": "x"}))


class TestMCPToolEnforcement(unittest.TestCase):
    """The MCP tool is the authoritative check — it runs in its own process."""

    def test_tool_refuses_a_tampered_payload_without_calling_the_broker(self):
        import mcp_server
        tampered = json.loads(json.dumps(PAYLOAD))
        tampered["qty"] = "999"

        with patch("mcp_server.requests.post") as post:
            out = json.loads(mcp_server.dispatch_mleg_order(
                json.dumps(tampered), canonical_payload_hash(PAYLOAD)))

        self.assertEqual(out["status"], "refused")
        post.assert_not_called()          # nothing reached the broker

    def test_tool_refuses_when_no_hash_is_supplied(self):
        import mcp_server
        with patch("mcp_server.requests.post") as post:
            out = json.loads(mcp_server.dispatch_mleg_order(json.dumps(PAYLOAD), ""))
        self.assertEqual(out["status"], "refused")
        post.assert_not_called()

    def test_tool_forwards_an_untouched_payload(self):
        import mcp_server
        with patch("mcp_server.requests.post") as post:
            post.return_value = MagicMock(status_code=200,
                                          json=lambda: {"id": "order-1"})
            out = json.loads(mcp_server.dispatch_mleg_order(
                json.dumps(PAYLOAD), canonical_payload_hash(PAYLOAD)))
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["order_id"], "order-1")
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMCPServerSpawn(unittest.TestCase):
    """The MCP server must run under the interpreter running the agent.

    `command="python3"` resolves through PATH. Inside a virtualenv that finds
    the system Python, which lacks this project's dependencies, so the server
    dies on `import mcp` and the client reports only "Connection closed".
    """

    def test_spawns_the_current_interpreter_not_a_path_lookup(self):
        import sys as _sys
        import inspect
        from ai_agent import options_agent
        src = inspect.getsource(options_agent._evaluate_and_dispatch) \
            if hasattr(options_agent, "_evaluate_and_dispatch") \
            else inspect.getsource(options_agent.OptionsExecutionAgent._evaluate_and_dispatch)
        self.assertIn("command=sys.executable", src)
        self.assertNotIn('command="python3"', src)

    def test_the_interpreter_can_import_the_server_dependencies(self):
        """Guards the real failure: the spawned interpreter lacking mcp."""
        import importlib
        for mod in ("mcp", "alpaca", "requests"):
            self.assertIsNotNone(importlib.import_module(mod))
