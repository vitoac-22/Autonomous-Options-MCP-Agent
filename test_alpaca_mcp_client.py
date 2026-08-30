"""
Tests for data_ingestion.alpaca_mcp_client.

The hackathon requires Alpaca's own MCP server or CLI. Our dispatch tool is a
server we wrote, which is defensible but arguable, so the read path now goes
through Alpaca's official server (`uvx alpaca-mcp-server`).

Two obstacles made the obvious approach unworkable, both verified by hand:

1. `alpaca-mcp-server` 2.3.0 depends on `mcp` 1.29.x, while our own server needs
   `mcp` 2.1.1 for `mcp.server.mcpserver.MCPServer`. Installing one breaks the
   other. Running Alpaca's under `uvx` keeps it in an isolated environment, so
   only the stdio wire is shared.

2. Even isolated, the `mcp` 2.1.1 *client* fails against it. The client offers
   protocol 2026-07-28; the server negotiates down to 2025-11-25 in its reply;
   the client rejects the downgrade and closes. A hand-written JSON-RPC
   handshake against the same server succeeds every time.

So this is a small stdio JSON-RPC client that accepts the negotiated version.
The framing is pure and tested here; the subprocess is a thin shell.
"""
import json
import unittest

from data_ingestion.alpaca_mcp_client import (
    encode_request, decode_response, parse_tool_result,
    initialize_params, MCPProtocolError, PROTOCOL_VERSION,
)


class TestFraming(unittest.TestCase):
    def test_request_is_one_json_line(self):
        raw = encode_request(1, "tools/list", {})
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(raw.count("\n"), 1)
        msg = json.loads(raw)
        self.assertEqual(msg["jsonrpc"], "2.0")
        self.assertEqual(msg["id"], 1)
        self.assertEqual(msg["method"], "tools/list")

    def test_notification_carries_no_id(self):
        msg = json.loads(encode_request(None, "notifications/initialized", None))
        self.assertNotIn("id", msg)

    def test_params_are_omitted_when_absent(self):
        self.assertNotIn("params", json.loads(encode_request(1, "ping", None)))

    def test_decode_returns_the_result_payload(self):
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
        self.assertEqual(decode_response(line, 1), {"tools": []})

    def test_decode_raises_on_a_jsonrpc_error(self):
        line = json.dumps({"jsonrpc": "2.0", "id": 1,
                           "error": {"code": -32601, "message": "No such method"}})
        with self.assertRaises(MCPProtocolError) as ctx:
            decode_response(line, 1)
        self.assertIn("No such method", str(ctx.exception))

    def test_decode_raises_on_an_id_mismatch(self):
        """A stale reply must not be read as the answer to this request."""
        line = json.dumps({"jsonrpc": "2.0", "id": 99, "result": {}})
        with self.assertRaises(MCPProtocolError):
            decode_response(line, 1)

    def test_decode_raises_on_non_json(self):
        with self.assertRaises(MCPProtocolError):
            decode_response("FastMCP banner text", 1)


class TestInitializeParams(unittest.TestCase):
    def test_offers_a_version_the_alpaca_server_accepts(self):
        """Verified by hand against alpaca-mcp-server 3.4.7: it accepts
        2024-11-05, 2025-06-18 and 2025-11-25."""
        self.assertIn(PROTOCOL_VERSION,
                      {"2024-11-05", "2025-06-18", "2025-11-25"})

    def test_params_have_the_required_handshake_fields(self):
        p = initialize_params()
        self.assertEqual(p["protocolVersion"], PROTOCOL_VERSION)
        self.assertIn("capabilities", p)
        self.assertIn("clientInfo", p)
        self.assertIn("name", p["clientInfo"])


class TestParseToolResult(unittest.TestCase):
    def test_extracts_text_content(self):
        payload = {"content": [{"type": "text", "text": "hello"}]}
        self.assertEqual(parse_tool_result(payload), "hello")

    def test_joins_multiple_text_blocks(self):
        payload = {"content": [{"type": "text", "text": "a"},
                               {"type": "text", "text": "b"}]}
        self.assertEqual(parse_tool_result(payload), "a\nb")

    def test_ignores_non_text_blocks(self):
        payload = {"content": [{"type": "image", "data": "..."},
                               {"type": "text", "text": "only this"}]}
        self.assertEqual(parse_tool_result(payload), "only this")

    def test_raises_when_the_tool_reports_an_error(self):
        """isError must not be read as a successful result."""
        payload = {"isError": True,
                   "content": [{"type": "text", "text": "symbol not found"}]}
        with self.assertRaises(MCPProtocolError) as ctx:
            parse_tool_result(payload)
        self.assertIn("symbol not found", str(ctx.exception))

    def test_empty_content_yields_empty_string(self):
        self.assertEqual(parse_tool_result({"content": []}), "")



class TestSnapshotAdapter(unittest.TestCase):
    """Alpaca's MCP server returns snapshots in wire shape (bp/ap, camelCase).
    build_leg expects SDK-shaped objects, so the adapter bridges them."""

    WIRE = {
        "_alpaca_mcp_security": {"trust": "untrusted_tool_output"},
        "data": {"snapshots": {"SPY260903C00782000": {
            "greeks": {"delta": 0.0576, "gamma": 0.0149,
                       "theta": -0.1161, "vega": 0.0929},
            "impliedVolatility": 0.0962,
            "latestQuote": {"bp": 0.18, "ap": 0.20},
            "dailyBar": {"v": 1128},
            "openInterest": 4210,
        }}},
    }

    def test_unwraps_the_security_envelope_and_data_key(self):
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        snaps = parse_snapshots(json.dumps(self.WIRE))
        self.assertIn("SPY260903C00782000", snaps)

    def test_exposes_greeks_in_the_shape_build_leg_expects(self):
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        s = parse_snapshots(json.dumps(self.WIRE))["SPY260903C00782000"]
        self.assertAlmostEqual(s.greeks.delta, 0.0576)

    def test_maps_wire_quote_keys_to_sdk_names(self):
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        s = parse_snapshots(json.dumps(self.WIRE))["SPY260903C00782000"]
        self.assertAlmostEqual(s.latest_quote.bid_price, 0.18)
        self.assertAlmostEqual(s.latest_quote.ask_price, 0.20)

    def test_carries_iv_liquidity_and_volume(self):
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        s = parse_snapshots(json.dumps(self.WIRE))["SPY260903C00782000"]
        self.assertAlmostEqual(s.implied_volatility, 0.0962)
        self.assertEqual(s.open_interest, 4210)
        self.assertEqual(s.volume, 1128)

    def test_result_feeds_build_leg_directly(self):
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        from data_ingestion.options_market_data import build_leg
        from quant_core.risk_gates import Leg
        s = parse_snapshots(json.dumps(self.WIRE))["SPY260903C00782000"]
        leg = build_leg("SPY260903C00782000", "sell", s, ratio=1)
        self.assertIsInstance(leg, Leg)
        self.assertAlmostEqual(leg.delta, 0.0576)
        self.assertAlmostEqual(leg.mid, 0.19)

    def test_missing_greeks_surface_as_none_not_zero(self):
        """A zero delta would read as a valid neutral leg; None makes build_leg
        raise, which is the behaviour that stops a trade on bad data."""
        from data_ingestion.alpaca_mcp_client import parse_snapshots
        wire = json.loads(json.dumps(self.WIRE))
        del wire["data"]["snapshots"]["SPY260903C00782000"]["greeks"]
        s = parse_snapshots(json.dumps(wire))["SPY260903C00782000"]
        self.assertIsNone(s.greeks)

if __name__ == "__main__":
    unittest.main(verbosity=2)
