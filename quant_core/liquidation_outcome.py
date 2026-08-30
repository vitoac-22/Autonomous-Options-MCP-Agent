"""
Did the emergency liquidation actually work?

`liquidate_portfolio` returns the symbols it could not close. The pipeline
discarded that value, so CI run 33322092186 reported success while the log
said:

    LIQUIDATION INCOMPLETE — still exposed: ['SPY260831P00675000']

An emergency exit failed, the account stayed exposed, and the build was green.
Same shape as the swallowed exception and the discarded broker rejection before
it: a failure that reports success.

One distinction is worth encoding rather than failing bluntly. Alpaca rejects
option market orders outside market hours, so a weekend or overnight run cannot
close anything — that is "cannot act yet", not an alarm, and failing every such
run would train everyone to ignore a red icon. During market hours the same
outcome means we could have closed the position and did not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LiquidationOutcome:
    ok: bool
    exit_code: int
    reason: str


def assess_liquidation(failures: Sequence[str], market_open: bool) -> LiquidationOutcome:
    """Judge a liquidation attempt. Pure: no clock, no network."""
    failures = list(failures)

    if not failures:
        return LiquidationOutcome(True, 0, "All positions closed.")

    symbols = ", ".join(failures)

    if not market_open:
        # Expected: Alpaca only accepts option market orders during RTH.
        return LiquidationOutcome(
            ok=False,
            exit_code=0,
            reason=(f"Still exposed: {symbols}. The market is closed, so these "
                    "cannot be closed until the next open — retrying then."),
        )

    return LiquidationOutcome(
        ok=False,
        exit_code=1,
        reason=(f"LIQUIDATION FAILED during market hours. Still exposed: {symbols}. "
                "The exit engine required these closed and the broker did not "
                "accept the orders."),
    )
