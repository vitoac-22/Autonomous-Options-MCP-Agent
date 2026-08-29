"""
Risk gates — the last check before an order reaches the broker.

`evaluate(proposal, snapshot) -> Verdict` is a PURE function. It performs no
network calls, invokes no model, and never reads the clock: everything it needs,
including the current time and whether the market is open, arrives on the
snapshot. Same inputs always produce the same verdict.

That purity is the point. It makes the layer exhaustively unit-testable, and it
is what lets an LLM propose trades without being trusted to place them. The
model chooses *what*; these gates decide *whether* and *how large*.

Sizing lives here deliberately. A proposal never carries a contract count — the
gates derive it from the sleeve budget and the structure's real maximum loss.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


# ── veto reasons ────────────────────────────────────────────────────────────
class VetoReason(str):
    """String subclass so reasons render readably in logs and on the dashboard."""
    UNDEFINED_RISK = "undefined_risk"
    WIDE_SPREAD = "wide_spread"
    ILLIQUID = "illiquid"
    DTE_OUT_OF_RANGE = "dte_out_of_range"
    NOT_DELTA_NEUTRAL = "not_delta_neutral"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_FLOOR = "drawdown_floor"
    MARKET_CLOSED = "market_closed"
    TOO_LATE_IN_SESSION = "too_late_in_session"
    TOO_MANY_POSITIONS = "too_many_positions"
    CONCENTRATED = "concentrated"
    SLEEVE_BUDGET_EXHAUSTED = "sleeve_budget_exhausted"
    DUPLICATE = "duplicate"
    NO_CAPACITY = "no_capacity"


# ── inputs ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Leg:
    symbol: str
    kind: str                 # "call" | "put"
    side: str                 # "buy" | "sell"
    strike: float
    ratio: int                # relative size within the structure
    delta: float              # REAL delta from get_option_snapshot, not a proxy
    bid: float
    ask: float
    open_interest: int
    volume: int
    expiry: date

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def position_sign(self) -> int:
        """+1 long, -1 short. Distinct from the sign of the leg's delta."""
        return 1 if self.side.lower() == "buy" else -1


@dataclass(frozen=True)
class TradeProposal:
    underlying: str
    structure: str            # "iron_condor" | "long_straddle" | ...
    sleeve: str               # "core" | "convex"
    legs: tuple[Leg, ...]
    rationale: str = ""


@dataclass(frozen=True)
class Position:
    underlying: str
    risk: float               # capital currently at risk, in dollars
    sleeve: str


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    starting_equity: float
    day_pnl: float
    now: datetime             # passed in — the gates never read the clock
    market_open: bool
    minutes_to_close: int
    open_positions: tuple[Position, ...] = ()
    submitted_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GateConfig:
    core_fraction: float = 0.60
    convex_fraction: float = 0.40
    max_concentration: float = 0.25      # per underlying, of equity
    max_spread_pct: float = 0.10         # (ask-bid)/mid on any leg
    min_open_interest: int = 500
    min_volume: int = 100
    min_dte: int = 1
    max_dte: int = 10
    max_abs_net_delta: float = 0.15      # per structure
    daily_loss_fraction: float = 0.06    # of starting equity
    drawdown_floor_fraction: float = 0.85
    max_open_positions: int = 8
    max_contracts: int = 50              # absolute sanity cap


@dataclass(frozen=True)
class Verdict:
    approved: bool
    contracts: int
    reasons: tuple[str, ...]
    idempotency_key: str
    max_loss_per_contract: Optional[float] = None
    net_delta: Optional[float] = None


# ── structure analysis (pure helpers) ───────────────────────────────────────
def _is_defined_risk(legs: tuple[Leg, ...]) -> bool:
    """Every short contract must be matched by a long contract of the same kind.

    A long option caps the loss regardless of its strike relative to the short:
    long 100 / short 110 calls is a bull call spread, not a naked short. So the
    test is on counts, not on strike ordering. What the strikes affect is how
    large the bounded loss is, which `_max_loss_per_contract` computes.
    """
    for kind in ("call", "put"):
        side = [leg for leg in legs if leg.kind.lower() == kind]
        shorts = sum(leg.ratio for leg in side if leg.position_sign < 0)
        longs = sum(leg.ratio for leg in side if leg.position_sign > 0)
        if shorts > longs:
            return False
    return True


def _net_premium_per_contract(legs: tuple[Leg, ...]) -> float:
    """Positive = net credit received, negative = net debit paid."""
    return sum(leg.mid * leg.ratio * 100.0 * -leg.position_sign for leg in legs)


def _widest_spread(legs: tuple[Leg, ...]) -> Optional[float]:
    """Widest strike gap among legs of the same kind — the loss cap on a credit
    structure."""
    widths = []
    for kind in ("call", "put"):
        strikes = [leg.strike for leg in legs if leg.kind.lower() == kind]
        if len(strikes) >= 2:
            widths.append(max(strikes) - min(strikes))
    return max(widths) if widths else None


def _max_loss_per_contract(legs: tuple[Leg, ...]) -> Optional[float]:
    """Worst case for one unit of the structure. None means unbounded."""
    if not _is_defined_risk(legs):
        return None

    net = _net_premium_per_contract(legs)
    if net < 0:
        return -net                       # debit structure: you can lose the premium

    width = _widest_spread(legs)
    if width is None:
        return None                       # credit taken with nothing defining the loss
    loss = width * 100.0 - net
    return max(loss, 1.0)                 # guard against a degenerate zero denominator


def _net_delta_per_contract(legs: tuple[Leg, ...]) -> float:
    return sum(leg.delta * leg.ratio * leg.position_sign for leg in legs)


def _idempotency_key(proposal: TradeProposal, snapshot: AccountSnapshot) -> str:
    """Stable for the same structure on the same trading day, so a retried or
    duplicated pipeline run cannot double-submit."""
    parts = [
        snapshot.now.date().isoformat(),
        proposal.underlying,
        proposal.structure,
        proposal.sleeve,
        *sorted(f"{leg.symbol}:{leg.side}:{leg.ratio}" for leg in proposal.legs),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


# ── the gate ────────────────────────────────────────────────────────────────
def evaluate(proposal: TradeProposal,
             snapshot: AccountSnapshot,
             config: GateConfig = GateConfig()) -> Verdict:
    """Approve or veto a proposal, and size it. Pure — no I/O, no clock reads.

    Every applicable gate runs; reasons accumulate rather than short-circuiting,
    so the dashboard can show every reason a trade was declined instead of only
    the first one hit.
    """
    reasons: list[str] = []
    legs = proposal.legs
    key = _idempotency_key(proposal, snapshot)

    # ── account-level state ────────────────────────────────────────────────
    if not snapshot.market_open:
        reasons.append(VetoReason.MARKET_CLOSED)
    if snapshot.minutes_to_close < 20:
        reasons.append(VetoReason.TOO_LATE_IN_SESSION)
    if snapshot.day_pnl <= -abs(snapshot.starting_equity * config.daily_loss_fraction):
        reasons.append(VetoReason.DAILY_LOSS_LIMIT)
    if key in snapshot.submitted_keys:
        reasons.append(VetoReason.DUPLICATE)
    if len(snapshot.open_positions) >= config.max_open_positions:
        reasons.append(VetoReason.TOO_MANY_POSITIONS)

    # The convex sleeve is the first thing to switch off in a drawdown; the
    # defined-risk core keeps running.
    floor = snapshot.starting_equity * config.drawdown_floor_fraction
    if proposal.sleeve == "convex" and snapshot.equity < floor:
        reasons.append(VetoReason.DRAWDOWN_FLOOR)

    # ── per-leg quality ────────────────────────────────────────────────────
    for leg in legs:
        if leg.mid <= 0:
            reasons.append(VetoReason.ILLIQUID)          # also avoids /0 below
            continue
        if (leg.ask - leg.bid) / leg.mid > config.max_spread_pct:
            reasons.append(VetoReason.WIDE_SPREAD)
        if leg.open_interest < config.min_open_interest or leg.volume < config.min_volume:
            reasons.append(VetoReason.ILLIQUID)
        dte = (leg.expiry - snapshot.now.date()).days
        if dte < config.min_dte or dte > config.max_dte:
            reasons.append(VetoReason.DTE_OUT_OF_RANGE)

    # ── structure shape ────────────────────────────────────────────────────
    max_loss = _max_loss_per_contract(legs)
    if max_loss is None:
        reasons.append(VetoReason.UNDEFINED_RISK)

    net_delta = _net_delta_per_contract(legs)
    if abs(net_delta) > config.max_abs_net_delta:
        reasons.append(VetoReason.NOT_DELTA_NEUTRAL)

    # ── budget and sizing ──────────────────────────────────────────────────
    contracts = 0
    if max_loss is not None:
        fraction = config.core_fraction if proposal.sleeve == "core" else config.convex_fraction
        sleeve_used = sum(p.risk for p in snapshot.open_positions if p.sleeve == proposal.sleeve)
        sleeve_room = snapshot.equity * fraction - sleeve_used

        underlying_used = sum(p.risk for p in snapshot.open_positions
                              if p.underlying == proposal.underlying)
        concentration_room = snapshot.equity * config.max_concentration - underlying_used

        if sleeve_room <= 0:
            reasons.append(VetoReason.SLEEVE_BUDGET_EXHAUSTED)
        if concentration_room <= 0:
            reasons.append(VetoReason.CONCENTRATED)

        budget = min(sleeve_room, concentration_room)
        if budget > 0:
            contracts = min(int(budget // max_loss), config.max_contracts)
            if contracts < 1:
                reasons.append(VetoReason.NO_CAPACITY)
                contracts = 0

    # Deduplicate while preserving the order gates ran in.
    unique = tuple(dict.fromkeys(reasons))
    approved = not unique and contracts >= 1

    return Verdict(
        approved=approved,
        contracts=contracts if approved else 0,
        reasons=unique,
        idempotency_key=key,
        max_loss_per_contract=max_loss,
        net_delta=net_delta,
    )
