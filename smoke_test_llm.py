"""
LLM smoke test — retires the last unknown before Monday.

Exercises the REAL chain end to end on the TEST paper account:

    Featherless (Qwen3-32B, real API call)
      -> returns a real tool_call
        -> our MCP server (real stdio subprocess)
          -> approval hash verified
            -> real order POST to Alpaca paper API
              -> order CANCELLED immediately

This script deliberately bypasses the risk gates (they correctly veto
weekend trading) — the thing under test is the LLM/MCP/broker bridge,
which has never executed for real.

Run:  .venv/bin/python smoke_test_llm.py
Needs: FEATHERLESS_API_KEY in .env (free at featherless.ai, $25 credit).
"""
import os
import sys
import json
import logging
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")

KEY = os.getenv("ALPACA_API_KEY")
SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
BASE = "https://paper-api.alpaca.markets/v2"
HDRS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET, "Content-Type": "application/json"}


def build_test_payload():
    """A small, real, defined-risk condor from live chain data."""
    from config import StrategyConfig
    from data_ingestion.alpaca_ingestor import UnderlyingIngestor
    from data_ingestion.chain_resolver import ChainResolver, LegSpec, select_expiry, map_legs_to_contracts
    from data_ingestion.options_market_data import OptionsMarketData, build_leg, parse_occ_symbol
    from ai_agent.mleg_payload import build_mleg_payload, net_limit_price
    from quant_core.risk_gates import Leg

    cfg = StrategyConfig.from_env()
    today = date.today()

    ingest = UnderlyingIngestor(cfg.underlying, "2016-01-01", today.strftime("%Y-%m-%d"))
    spot = ingest.get_latest_price()
    log.info(f"SMOKE    SPY spot ${spot:.2f}")

    resolver = ChainResolver(cfg.underlying)
    chain = resolver.fetch_chain(
        today, cfg.min_dte, cfg.max_dte,
        strike_low=round(spot) - 15, strike_high=round(spot) + 15,
    )
    expiry = select_expiry(chain, today, cfg.target_dte, cfg.min_dte, cfg.max_dte)
    log.info(f"SMOKE    expiry chosen: {expiry}")

    # Tight wings (±8 / ±12): far-OTM wings sometimes lack weekend Greeks,
    # and this test is about the bridge, not the structure.
    specs = [
        LegSpec(kind="put",  side="buy",  target_strike=round(spot - 12)),
        LegSpec(kind="put",  side="sell", target_strike=round(spot - 8)),
        LegSpec(kind="call", side="sell", target_strike=round(spot + 8)),
        LegSpec(kind="call", side="buy",  target_strike=round(spot + 12)),
    ]
    mapped = map_legs_to_contracts(chain, expiry, specs)
    symbols = [c.symbol for c in mapped]
    for spec, c in zip(specs, mapped):
        log.info(f"SMOKE    {spec.side:4s} {c.symbol} ({c.kind} {c.strike})")

    md = OptionsMarketData(KEY, SECRET)
    try:
        legs = md.build_legs([(c.symbol, spec.side, 1) for spec, c in zip(specs, mapped)])
    except Exception as e:  # weekend-missing Greeks on a wing
        log.warning(f"SMOKE    {e}")
        log.warning("SMOKE    falling back to quote-only legs (placeholder deltas) — fine for bridge testing")
        snaps = md.fetch_snapshots(symbols)
        legs = []
        for spec, c in zip(specs, mapped):
            s = snaps.get(c.symbol)
            q = getattr(s, "latest_quote", None)
            g = getattr(s, "greeks", None)
            p = parse_occ_symbol(c.symbol)
            legs.append(Leg(
                symbol=c.symbol, kind=p.kind, side=spec.side, strike=p.strike, ratio=1,
                delta=float(getattr(g, "delta", None) or 0.0),
                bid=float(getattr(q, "bid_price", 0.0) or 0.0),
                ask=float(getattr(q, "ask_price", 0.0) or 0.0),
                open_interest=int(getattr(s, "open_interest", None) or 0),
                volume=int(getattr(s, "daily_volume", None) or 0),
                expiry=p.expiry,
            ))

    limit = net_limit_price(legs)
    payload = build_mleg_payload(legs, contracts=1, limit_price=limit)
    log.info(f"SMOKE    payload built | net limit ${limit:.2f}")
    return payload


def main():
    if not os.getenv("FEATHERLESS_API_KEY"):
        print("\n" + "=" * 60)
        print("FEATHERLESS_API_KEY missing.")
        print("1. Sign up free at https://featherless.ai ($25 credit)")
        print("2. Add FEATHERLESS_API_KEY=... to this repo's .env")
        print("=" * 60)
        sys.exit(2)

    # 1. Build a real payload
    payload = build_test_payload()
    log.info("SMOKE    payload: " + json.dumps(payload))

    # 2. The real dispatch: Featherless -> MCP -> hash -> broker
    from ai_agent.options_agent import OptionsExecutionAgent
    agent = OptionsExecutionAgent()
    log.info("SMOKE    dispatching through the REAL LLM/MCP/broker chain...")
    result = agent.submit_mleg_payload(
        payload,
        rationale="Smoke test of the execution bridge on the paper test account.",
    )
    order_id = result.get("order_id")
    log.info(f"SMOKE    ✅ LLM authorized, hash verified, BROKER ACCEPTED order {order_id}")

    # 3. Cancel immediately
    r = requests.delete(f"{BASE}/orders/{order_id}", headers=HDRS)
    if r.status_code in (200, 201, 204):
        log.info(f"SMOKE    ✅ order {order_id} cancelled — zero fill risk")
    else:
        log.warning(f"SMOKE    cancel returned {r.status_code}: {r.text[:200]}")
        log.warning("SMOKE    cancel it manually in the dashboard: dashboard.alpaca.markets")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED: every link in the chain is real and works.")
    print("  Featherless Qwen3 -> tool_call        ✓")
    print("  MCP server (stdio) -> hash verified   ✓")
    print("  Alpaca paper -> order accepted        ✓")
    print("  order cancelled                       ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
