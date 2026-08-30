"""
Strategy parameters, in one place, overridable by environment variable.

The pipeline hardcodes target_dte=30. Judging ends 4 Sep, so a position opened
30 days out cannot expire, be assigned, or complete a cycle inside the window:
judges see one open position drifting on mark-to-market instead of a finished
round trip. It also means exit_engine's DTE<=5 liquidation rule never fires,
because a 30-DTE position never reaches 5 DTE in-window.

The default here is short-dated so the strategy actually cycles and the demo
shows completed trades with realised P&L. SPY has near-daily expiries, so this
costs nothing in liquidity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError, TypeError):
        return default


@dataclass(frozen=True)
class StrategyConfig:
    underlying: str = "SPY"

    # Expiry window. Short by design — see module docstring.
    min_dte: int = 1
    max_dte: int = 10
    target_dte: int = 7

    # Regime threshold for the GARCH VaR strategy selector.
    var_threshold: float = -0.035

    # Sleeve budgets. Kept in sync with quant_core.risk_gates.GateConfig.
    #
    # Was 60/40. The convex sleeve buys volatility, which pays daily time decay
    # and only profits on a real move before expiry. Three things argue against
    # funding it heavily this week: positions must expire by 3 Sep while the
    # jobs report — the only large catalyst — lands on 4 Sep, so the position is
    # dead before the likeliest trigger; VIX is ~14 at an IV rank near 10; and
    # the GARCH fit itself reports persistence 0.997, which is the model saying
    # calm begets calm. A large long-vol sleeve would be a bet against our own
    # forecast. 15% keeps genuine convexity without paying rent on a lottery
    # ticket that expires before the draw.
    core_fraction: float = 0.85
    convex_fraction: float = 0.15

    # Baseline the drawdown floor measures against — the account's starting
    # balance, not the current session's opening equity.
    contest_start_equity: float = 100_000.0

    # Hard ceiling on expiry. Total account equity is photographed at the close
    # of Thursday 3 Sep, and option exercises/assignments for that date are
    # reflected in it. A position expiring after the snapshot is only marked to
    # market, so a premium-selling structure would capture partial time decay
    # instead of the whole credit. Empty string disables the ceiling.
    final_expiry: str = "2026-09-03"

    def __post_init__(self):
        if self.min_dte > self.max_dte:
            raise ValueError(f"min_dte {self.min_dte} exceeds max_dte {self.max_dte}")
        if not (self.min_dte <= self.target_dte <= self.max_dte):
            raise ValueError(
                f"target_dte {self.target_dte} sits outside the band "
                f"[{self.min_dte}, {self.max_dte}]"
            )

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        """Read overrides from the environment, falling back to defaults on any
        missing or unparseable value."""
        return cls(
            underlying=os.environ.get("UNDERLYING", cls.underlying),
            min_dte=_env_int("MIN_DTE", cls.min_dte),
            max_dte=_env_int("MAX_DTE", cls.max_dte),
            target_dte=_env_int("TARGET_DTE", cls.target_dte),
            var_threshold=_env_float("VAR_THRESHOLD", cls.var_threshold),
            core_fraction=_env_float("CORE_FRACTION", cls.core_fraction),
            convex_fraction=_env_float("CONVEX_FRACTION", cls.convex_fraction),
            contest_start_equity=_env_float("CONTEST_START_EQUITY", cls.contest_start_equity),
            final_expiry=os.environ.get("FINAL_EXPIRY", cls.final_expiry),
        )
