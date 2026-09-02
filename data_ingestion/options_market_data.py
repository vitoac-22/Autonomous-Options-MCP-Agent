"""
Real option market data — Greeks, implied volatility, quotes, liquidity.

Replaces the invented inputs in main_pipeline:

    d_proxy = 0.1 if abs(leg['sign']) == 1 else 0.5
    precios_estimados.append(2.5)

`sign` is always +/-1, so the conditional always yielded 0.1 and the `0.5`
branch was unreachable. Every leg therefore carried the same fabricated delta
and a flat $2.50 price, which meant DeltaNeutralOptimizer was minimising net
delta over constants and sizing against a fictional capital constraint.

Alpaca returns the real values from `get_option_snapshot`, which this codebase
never called. Fetching them also unlocks the comparison the strategy is
supposed to rest on: implied volatility from the market against the GARCH
forecast of realised volatility.

Network access and interpretation are kept apart. `parse_occ_symbol` and
`build_leg` are pure and fully tested; `OptionsMarketData` is a thin shell over
the SDK.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from quant_core.risk_gates import Leg


class MissingGreeksError(RuntimeError):
    """Raised when a snapshot lacks Greeks or a quote.

    Deliberately fatal. Substituting a default delta is precisely the bug this
    module exists to remove — a missing value must stop the trade, not quietly
    become 0.1.
    """


# ── OCC symbol parsing ──────────────────────────────────────────────────────
# Format: <ROOT><YYMMDD><C|P><STRIKE * 1000, 8 digits>
#   SPY260904C00105000 -> SPY, 2026-09-04, call, 105.0
_OCC = re.compile(r"^(?P<root>[A-Z]{1,6}?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OCCSymbol:
    underlying: str
    expiry: date
    kind: str          # "call" | "put"
    strike: float


def parse_occ_symbol(symbol: str) -> OCCSymbol:
    """Parse an OCC option symbol.

    Handles roots of any length. `exit_engine` currently slices `symbol[3:9]`
    for the date, which silently breaks on any underlying that is not exactly
    three characters.
    """
    m = _OCC.match(symbol.strip().upper())
    if not m:
        raise ValueError(f"Not a valid OCC option symbol: {symbol!r}")

    ymd = m.group("ymd")
    try:
        expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    except ValueError as exc:
        raise ValueError(f"Invalid expiry in symbol {symbol!r}") from exc

    return OCCSymbol(
        underlying=m.group("root"),
        expiry=expiry,
        kind="call" if m.group("cp") == "C" else "put",
        strike=int(m.group("strike")) / 1000.0,
    )


# ── snapshot -> Leg (pure) ──────────────────────────────────────────────────
def build_leg(symbol: str, side: str, snapshot, ratio: int = 1) -> Leg:
    """Turn an Alpaca option snapshot into a Leg the risk gates can evaluate.

    Pure: `snapshot` is whatever the caller already fetched. Raises rather than
    defaulting when Greeks or a quote are absent.
    """
    parsed = parse_occ_symbol(symbol)

    greeks = getattr(snapshot, "greeks", None)
    if greeks is None or getattr(greeks, "delta", None) is None:
        raise MissingGreeksError(
            f"No Greeks for {symbol}. Refusing to substitute a default delta — "
            "that is the bug this module replaces."
        )

    quote = getattr(snapshot, "latest_quote", None)
    if quote is None or quote.bid_price is None or quote.ask_price is None:
        raise MissingGreeksError(f"No quote for {symbol}; cannot price the leg.")

    # Absent liquidity data must read as illiquid so the gates veto, never as
    # unlimited depth.
    return Leg(
        symbol=symbol,
        kind=parsed.kind,
        side=side.lower(),
        strike=parsed.strike,
        ratio=int(ratio),
        delta=float(greeks.delta),
        bid=float(quote.bid_price),
        ask=float(quote.ask_price),
        open_interest=int(getattr(snapshot, "open_interest", None) or 0),
        volume=int(getattr(snapshot, "volume", None) or 0),
        expiry=parsed.expiry,
    )


def prefer_chain_liquidity(legs, contracts) -> tuple:
    """Fill zero open-interest from the contract chain.

    alpaca-py's OptionsSnapshot model carries no open_interest or volume
    fields at all (verified against 0.44.0: greeks, IV, quotes, trades only),
    so the SDK read path reports zeros regardless of feed — and the free
    indicative snapshot serves zeros during regular hours besides. The
    contract chain, fetched separately via get_option_contracts, carries
    real open interest. When a leg reports zero, take the chain's number; a
    real snapshot value always wins. Pure: same inputs, same legs.
    """
    from dataclasses import replace as _replace
    by_symbol = {c.symbol: c for c in contracts}
    out = []
    for leg in legs:
        chain = by_symbol.get(leg.symbol)
        if leg.open_interest == 0 and chain is not None and chain.open_interest:
            leg = _replace(leg, open_interest=int(chain.open_interest))
        out.append(leg)
    return tuple(out)


def implied_volatility(snapshot) -> Optional[float]:
    """ATM implied volatility, for comparison against the GARCH forecast."""
    iv = getattr(snapshot, "implied_volatility", None)
    return float(iv) if iv is not None else None


# ── the I/O shell ───────────────────────────────────────────────────────────
class OptionsMarketData:
    """Thin wrapper over Alpaca's option snapshot endpoint."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = (api_secret or os.getenv("ALPACA_SECRET_KEY")
                           or os.getenv("ALPACA_API_SECRET"))
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing Alpaca credentials in environment.")
        self._client = None

    def _get_client(self):
        # Imported lazily so the pure helpers above stay usable without the SDK.
        if self._client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._client = OptionHistoricalDataClient(self.api_key, self.api_secret)
        return self._client

    def fetch_snapshots(self, symbols: Iterable[str]) -> dict:
        from alpaca.data.requests import OptionSnapshotRequest
        symbols = list(symbols)
        if not symbols:
            return {}
        return self._get_client().get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=symbols)
        )

    def build_legs(self, specs: Iterable[tuple[str, str, int]]) -> list[Leg]:
        """specs: (symbol, side, ratio) -> Legs carrying real Greeks and quotes."""
        specs = list(specs)
        snapshots = self.fetch_snapshots(symbol for symbol, _, _ in specs)
        legs = []
        for symbol, side, ratio in specs:
            snap = snapshots.get(symbol)
            if snap is None:
                raise MissingGreeksError(f"Alpaca returned no snapshot for {symbol}.")
            legs.append(build_leg(symbol, side, snap, ratio))
        return legs


def build_legs_via_mcp(specs, client=None) -> list:
    """Legs built from snapshots read through **Alpaca's own MCP server**.

    Same output as OptionsMarketData.build_legs, sourced through Alpaca's
    official MCP server rather than the SDK. This is what makes the "MCP or CLI"
    requirement unambiguous: their server reads the Greeks, ours dispatches the
    order.

    specs: iterable of (symbol, side, ratio).
    """
    from data_ingestion.alpaca_mcp_client import AlpacaMCPClient, parse_snapshots, MCPProtocolError

    specs = list(specs)
    symbols = [symbol for symbol, _, _ in specs]

    def _fetch(mcp):
        return parse_snapshots(mcp.call("get_option_snapshot",
                                        {"symbols": ",".join(symbols)}))

    if client is not None:
        snapshots = _fetch(client)
    else:
        with AlpacaMCPClient() as mcp:
            snapshots = _fetch(mcp)

    legs = []
    for symbol, side, ratio in specs:
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            raise MissingGreeksError(
                f"Alpaca's MCP server returned no snapshot for {symbol}."
            )
        legs.append(build_leg(symbol, side, snapshot, ratio))
    return legs

def adapt_wings_to_feed(chain_contracts, specs, snapshot_fetch) -> list:
    """Move wings (and if needed shorts) to strikes the feed actually covers.

    On the free indicative tier, Greeks exist only where the quote is
    two-sided: past a bid of $0.00 coverage stops entirely (measured live:
    SPY 2026-09-03 calls had Greeks through C779 and none from C780 up). A
    wing out there is unpriced, and inventing its delta is the one thing
    this codebase refuses to do. So each credit side keeps its shape but
    slides inward to the nearest covered strikes, keeping at least one
    strike of width. `snapshot_fetch(symbols) -> {symbol: snapshot}` is
    injected so the helper stays testable.
    """
    def _has_greeks(sym):
        snap = fetched.get(sym)
        g = getattr(snap, "greeks", None) if snap is not None else None
        return g is not None and getattr(g, "delta", None) is not None

    def _strike(c):
        return c.strike

    # Probe every candidate strike; the snapshot endpoint caps at 100
    # symbols per call, so batch and merge.
    by_kind = {}
    for c in chain_contracts:
        by_kind.setdefault(c.kind, []).append(c)
    probe = [c.symbol for c in chain_contracts]
    fetched = {}
    for i in range(0, len(probe), 100):
        fetched.update(dict(snapshot_fetch(probe[i:i + 100]) or {}))

    adjusted = []
    for spec in specs:
        if isinstance(spec, dict):
            adjusted.append(dict(kind=spec["kind"], side=spec["side"],
                                 target_strike=float(spec["target_strike"])))
        else:
            adjusted.append(dict(kind=spec.kind, side=spec.side,
                                 target_strike=float(spec.target_strike)))

    for kind, inward in (("call", -1), ("put", +1)):
        side = [s for s in adjusted if s["kind"] == kind]
        if len(side) != 2:
            continue
        wing = next(s for s in side if s["side"] == "buy")
        short = next(s for s in side if s["side"] == "sell")
        # wing is further from spot than short; "inward" moves toward spot.
        strikes = sorted({c.strike for c in by_kind.get(kind, [])},
                         reverse=(inward < 0))
        if not strikes:
            continue
        # Candidates from the ideal wing inward, past the short by a margin.
        lo = min(wing["target_strike"], short["target_strike"]) - 2 * abs(inward)
        hi = max(wing["target_strike"], short["target_strike"]) + 2 * abs(inward)
        span = [k for k in strikes if lo - 1 <= k <= hi + 1]

        def _covered(k):
            sym = next((c.symbol for c in by_kind[kind] if c.strike == k), None)
            return sym is not None and _has_greeks(sym)

        # Slide the wing inward until covered.
        path = [k for k in span if (k - wing["target_strike"]) * inward >= 0]
        new_wing = None
        for k in sorted(path, key=lambda k: abs(k - wing["target_strike"])):
            if _covered(k):
                new_wing = k
                break
        if new_wing is None:
            continue
        wing["target_strike"] = new_wing
        # Keep at least one strike of width: slide the short inward too if the
        # wing overtook it.
        step = 1.0 if strikes and min(abs(a - b) for a in strikes[:2] for b in strikes[:2] if a != b) else 1.0
        if (new_wing - short["target_strike"]) * inward < 1:
            short_path = [k for k in span
                          if (k - new_wing) * inward >= 1 and _covered(k)]
            if short_path:
                short["target_strike"] = min(
                    short_path, key=lambda k: abs(k - short["target_strike"]))
    return adjusted
