"""
Multi-leg (mleg) order payload construction — pure, so it can be tested without
touching the network.

Fixes two defects in the current builder.

Missing required fields. Alpaca's mleg order class requires a top-level `qty`
and a `position_intent` on every leg. The existing payload carries neither, so
the order is rejected 422 and the atomic execution path has never successfully
placed an order.

Ratio truncation. The existing code computes

    base_qty = min(leg['qty'] for leg in legs)
    ratio_qty = int(leg['qty'] / base_qty)

Integer division truncates: [4,6,6,4] collapses to [1,1,1,1], quietly turning a
2:3:3:2 structure into an equal-ratio one with a different risk profile than the
optimiser computed. Reducing by the greatest common divisor instead is exact,
and it satisfies Alpaca's requirement that the ratio set be in lowest terms.

Numeric fields are cast to strings here. The current system prompt asks the
model to remember to do that; code that cannot forget is better than a prompt
that can be ignored.
"""
from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Iterable, Optional, Sequence

MAX_LEGS = 4  # Alpaca's limit for a multi-leg order


class PositionIntent:
    OPEN = "open"
    CLOSE = "close"


def _format_number(value: float) -> str:
    """Render without scientific notation or trailing zeros."""
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def _reduced_ratios(legs: Sequence) -> list[int]:
    """Leg ratios in lowest terms. Exact — never truncates."""
    ratios = [max(int(leg.ratio), 1) for leg in legs]
    divisor = reduce(gcd, ratios)
    return [r // divisor for r in ratios]


def _intent_for(side: str, intent: str) -> str:
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"Invalid side: {side!r}")
    suffix = "open" if intent == PositionIntent.OPEN else "close"
    return f"{side}_to_{suffix}"


def build_mleg_payload(legs: Iterable,
                       contracts: int,
                       limit_price: Optional[float] = None,
                       intent: str = PositionIntent.OPEN) -> dict:
    """Build the JSON body for POST /v2/orders with order_class="mleg".

    `contracts` is how many times to execute the whole structure — it comes from
    the risk gates, never from a model. Per-leg contract count is
    `contracts * ratio_qty`.

    A limit price is strongly preferred: a market order across four option legs
    with wide spreads pays the spread four times.
    """
    legs = tuple(legs)
    if not legs:
        raise ValueError("Cannot build an mleg order with no legs.")
    if len(legs) > MAX_LEGS:
        raise ValueError(f"Alpaca allows at most {MAX_LEGS} legs; got {len(legs)}.")
    if int(contracts) < 1:
        raise ValueError(f"contracts must be >= 1, got {contracts}.")

    ratios = _reduced_ratios(legs)

    payload = {
        "order_class": "mleg",
        "qty": str(int(contracts)),
        "type": "limit" if limit_price is not None else "market",
        "time_in_force": "day",          # options accept only day or gtc
        "legs": [
            {
                "symbol": leg.symbol,
                "ratio_qty": str(ratio),
                "side": leg.side.lower(),
                "position_intent": _intent_for(leg.side, intent),
            }
            for leg, ratio in zip(legs, ratios)
        ],
    }

    if limit_price is not None:
        if float(limit_price) <= 0:
            # A zero or negative limit is never a sane instruction to a broker:
            # on a credit structure it accepts no credit, on a debit one it can
            # never fill. Surfaces a bad quote rather than sending it.
            raise ValueError(
                f"limit_price must be positive, got {limit_price}. "
                "This usually means a leg is missing a real quote."
            )
        payload["limit_price"] = _format_number(float(limit_price))

    return payload


def net_limit_price(legs: Iterable, pad: float = 0.05) -> float:
    """Net debit/credit for one unit of the structure, from real mid prices.

    Returns a positive number: the absolute net premium, padded toward the side
    that fills. Requires legs carrying real quotes — see
    data_ingestion.options_market_data.
    """
    legs = tuple(legs)
    ratios = _reduced_ratios(legs)
    net = sum(leg.mid * ratio * -leg.position_sign
              for leg, ratio in zip(legs, ratios))
    magnitude = abs(net)
    return round(magnitude * (1 - pad) if net > 0 else magnitude * (1 + pad), 2)
