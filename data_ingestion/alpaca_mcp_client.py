"""
A minimal stdio JSON-RPC client for Alpaca's official MCP server.

The hackathon requires Alpaca's own MCP server or CLI. Our `dispatch_mleg_order`
runs on a server we wrote — real MCP, and a genuine differentiator, but it wraps
Alpaca's REST API rather than being theirs. Routing the read path through their
official server removes the ambiguity: theirs reads the chain and the Greeks,
ours dispatches the order.

Two obstacles ruled out the obvious approach, both verified by hand.

`alpaca-mcp-server` 2.3.0 depends on `mcp` 1.29.x, while our own server needs
`mcp` 2.1.1 for `mcp.server.mcpserver.MCPServer`. Installing one downgrades the
other and breaks it. Running Alpaca's server under `uvx` keeps it in its own
isolated environment, so the two only share the stdio wire — where the version
does not have to match.

Even isolated, the `mcp` 2.1.1 *client* cannot talk to it. The client offers
protocol 2026-07-28; the server negotiates down to 2025-11-25 in its reply; the
client rejects the downgrade and closes with an unhelpful "Connection closed".
A hand-written handshake against the same server succeeds every time, so this
speaks JSON-RPC directly and accepts the negotiated version.

The framing functions are pure and fully tested. The subprocess is a thin shell.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional

# Offered at handshake. Verified against alpaca-mcp-server 3.4.7, which accepts
# 2024-11-05, 2025-06-18 and 2025-11-25.
PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "delta-zero"


class MCPProtocolError(RuntimeError):
    """The server returned an error, or something that is not a valid reply."""


# ── pure framing ────────────────────────────────────────────────────────────
def encode_request(request_id: Optional[int], method: str,
                   params: Optional[dict]) -> str:
    """One JSON-RPC message per line. `request_id=None` makes a notification."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return json.dumps(message) + "\n"


def decode_response(line: str, expected_id: int) -> dict:
    """Return the `result` payload, or raise.

    An id mismatch raises rather than returning: a stale reply must never be
    read as the answer to the current request.
    """
    try:
        message = json.loads(line)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MCPProtocolError(f"Not JSON-RPC: {line[:200]!r}") from exc

    if "error" in message:
        err = message["error"]
        raise MCPProtocolError(f"{err.get('code')}: {err.get('message')}")

    if message.get("id") != expected_id:
        raise MCPProtocolError(
            f"Reply id {message.get('id')} does not match request {expected_id}."
        )
    return message.get("result", {})


def initialize_params() -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": CLIENT_NAME, "version": "1.0"},
    }


def parse_tool_result(result: dict) -> str:
    """Text content of a tool call. Raises when the tool reports an error."""
    blocks = result.get("content") or []
    text = "\n".join(b.get("text", "") for b in blocks
                     if b.get("type") == "text")
    if result.get("isError"):
        raise MCPProtocolError(f"Tool reported an error: {text or result}")
    return text


# ── the subprocess shell ────────────────────────────────────────────────────
class AlpacaMCPClient:
    """Alpaca's official MCP server, spoken to over stdio.

        with AlpacaMCPClient() as mcp:
            snapshot = mcp.call("get_option_snapshot", {"symbols": [...]})
    """

    def __init__(self, command: Optional[list[str]] = None,
                 env: Optional[dict] = None, timeout: float = 120.0):
        # uvx runs the server in its own environment, which is what keeps its
        # mcp 1.x from colliding with our mcp 2.x.
        self.command = command or ["uvx", "alpaca-mcp-server"]
        self.timeout = timeout
        self._env = {**os.environ, **(env or {})}
        self._env.setdefault("ALPACA_PAPER_TRADE", "true")
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0

    # context manager -------------------------------------------------------
    def __enter__(self) -> "AlpacaMCPClient":
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,          # FastMCP prints a banner here
            env=self._env, text=True, bufsize=1,
        )
        self._request("initialize", initialize_params())
        self._notify("notifications/initialized")
        return self

    def __exit__(self, *exc) -> bool:
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        return False

    # wire ------------------------------------------------------------------
    def _write(self, raw: str) -> None:
        if not self._proc or self._proc.poll() is not None:
            raise MCPProtocolError("Alpaca MCP server is not running.")
        self._proc.stdin.write(raw)
        self._proc.stdin.flush()

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._write(encode_request(request_id, method, params))

        line = self._proc.stdout.readline()
        if not line:
            raise MCPProtocolError(
                f"Alpaca MCP server closed the connection during {method}."
            )
        return decode_response(line.strip(), request_id)

    def _notify(self, method: str, params: Optional[dict] = None) -> None:
        self._write(encode_request(None, method, params))

    # api -------------------------------------------------------------------
    def list_tools(self) -> list[dict]:
        return self._request("tools/list", {}).get("tools", [])

    def call(self, name: str, arguments: dict) -> str:
        return parse_tool_result(
            self._request("tools/call", {"name": name, "arguments": arguments})
        )


# ── snapshot adapter ────────────────────────────────────────────────────────
#
# Alpaca's MCP server returns snapshots in wire shape — camelCase keys, `bp`/`ap`
# for bid/ask, and wrapped in a security envelope that marks the payload as
# untrusted tool output. `build_leg` expects SDK-shaped objects, so these small
# adapters bridge the two without either side knowing about the other.
class _Greeks:
    __slots__ = ("delta", "gamma", "theta", "vega", "rho")

    def __init__(self, raw: dict):
        self.delta = raw.get("delta")
        self.gamma = raw.get("gamma")
        self.theta = raw.get("theta")
        self.vega = raw.get("vega")
        self.rho = raw.get("rho")


class _Quote:
    __slots__ = ("bid_price", "ask_price", "bid_size", "ask_size")

    def __init__(self, raw: dict):
        self.bid_price = raw.get("bp")
        self.ask_price = raw.get("ap")
        self.bid_size = raw.get("bs")
        self.ask_size = raw.get("as")


class MCPSnapshot:
    """SDK-shaped view over one wire snapshot."""

    def __init__(self, raw: dict):
        greeks = raw.get("greeks")
        # None rather than an empty object: a zero delta would read as a valid
        # neutral leg, whereas None makes build_leg raise and stop the trade.
        self.greeks = _Greeks(greeks) if greeks else None
        quote = raw.get("latestQuote")
        self.latest_quote = _Quote(quote) if quote else None
        self.implied_volatility = raw.get("impliedVolatility")
        self.open_interest = raw.get("openInterest") or 0
        self.volume = (raw.get("dailyBar") or {}).get("v") or 0


def parse_snapshots(payload: str) -> dict[str, MCPSnapshot]:
    """Unwrap the tool response into {symbol: MCPSnapshot}."""
    try:
        body = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MCPProtocolError(f"Snapshot payload is not JSON: {payload[:200]!r}") from exc

    snapshots = ((body.get("data") or body).get("snapshots")) or {}
    return {symbol: MCPSnapshot(raw) for symbol, raw in snapshots.items()}
