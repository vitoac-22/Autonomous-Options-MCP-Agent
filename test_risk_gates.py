"""
Tests for quant_core.risk_gates.

The gate layer is a PURE function: (proposal, snapshot) -> Verdict. No network,
no LLM, no clock reads. That is what makes it exhaustively testable, and it is
the only thing standing between a model's output and a live order.

Every gate gets a test in both directions: one proposal it must approve, one it
must veto.
"""
import unittest
from datetime import date, datetime

from quant_core.risk_gates import (
    Leg, TradeProposal, Position, AccountSnapshot, GateConfig,
    evaluate, VetoReason,
)


# ── builders ────────────────────────────────────────────────────────────────
def leg(kind="call", side="sell", strike=100.0, ratio=1, delta=0.15,
        bid=1.00, ask=1.04, oi=2000, volume=500, expiry=date(2026, 9, 4)):
    return Leg(symbol=f"SPY{expiry:%y%m%d}{'C' if kind == 'call' else 'P'}{int(strike*1000):08d}",
               kind=kind, side=side, strike=strike, ratio=ratio, delta=delta,
               bid=bid, ask=ask, open_interest=oi, volume=volume, expiry=expiry)


def iron_condor(**kw):
    """Delta-neutral, defined-risk: long wings outside short bodies."""
    return (
        leg(kind="put",  side="buy",  strike=92.0, delta=-0.05, bid=0.40, ask=0.42, **kw),
        leg(kind="put",  side="sell", strike=95.0, delta=-0.16, bid=0.90, ask=0.94, **kw),
        leg(kind="call", side="sell", strike=105.0, delta=0.16, bid=0.90, ask=0.94, **kw),
        leg(kind="call", side="buy",  strike=108.0, delta=0.05, bid=0.40, ask=0.42, **kw),
    )


def naked_strangle():
    """Short both sides with NO protective wings — unbounded loss."""
    return (
        leg(kind="put",  side="sell", strike=95.0, delta=-0.16),
        leg(kind="call", side="sell", strike=105.0, delta=0.16),
    )


def proposal(legs=None, sleeve="core", contracts=1, underlying="SPY"):
    return TradeProposal(
        underlying=underlying,
        structure="iron_condor",
        sleeve=sleeve,
        legs=legs if legs is not None else iron_condor(),
        rationale="IV rich vs GARCH forecast",
    )


def snapshot(equity=100_000.0, day_pnl=0.0, positions=(), market_open=True,
             minutes_to_close=180, submitted=frozenset(), now=datetime(2026, 9, 1, 14, 0)):
    return AccountSnapshot(
        equity=equity,
        starting_equity=100_000.0,
        day_pnl=day_pnl,
        now=now,
        market_open=market_open,
        minutes_to_close=minutes_to_close,
        open_positions=positions,
        submitted_keys=submitted,
    )


class TestHappyPath(unittest.TestCase):
    def test_clean_condor_is_approved_and_sized(self):
        v = evaluate(proposal(), snapshot())
        self.assertTrue(v.approved, msg=f"unexpected veto: {v.reasons}")
        self.assertGreaterEqual(v.contracts, 1)
        self.assertEqual(v.reasons, ())

    def test_gates_size_the_position_not_the_caller(self):
        """Contract count is derived from the sleeve budget, never proposed."""
        small = evaluate(proposal(), snapshot(equity=20_000.0))
        large = evaluate(proposal(), snapshot(equity=400_000.0))
        self.assertLess(small.contracts, large.contracts)

    def test_verdict_is_deterministic(self):
        a = evaluate(proposal(), snapshot())
        b = evaluate(proposal(), snapshot())
        self.assertEqual((a.approved, a.contracts, a.reasons),
                         (b.approved, b.contracts, b.reasons))


class TestDefinedRisk(unittest.TestCase):
    def test_naked_short_is_vetoed(self):
        """The live run stranded a naked position. This is that guard."""
        v = evaluate(proposal(legs=naked_strangle()), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.UNDEFINED_RISK, v.reasons)

    def test_more_shorts_than_longs_on_one_side_is_naked(self):
        """Two short calls covered by only one long call leaves one uncovered."""
        legs = (leg(kind="call", side="sell", strike=105.0, ratio=2),
                leg(kind="call", side="buy",  strike=108.0, ratio=1))
        v = evaluate(proposal(legs=legs), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.UNDEFINED_RISK, v.reasons)

    def test_long_below_short_call_is_still_defined_risk(self):
        """A long call caps the loss at ANY strike — long 100 / short 110 is a
        bull call spread, not a naked short. Defined risk is about counts, not
        strike order."""
        legs = (leg(kind="call", side="buy",  strike=100.0, delta=0.55),
                leg(kind="call", side="sell", strike=110.0, delta=0.10))
        v = evaluate(proposal(legs=legs), snapshot())
        self.assertNotIn(VetoReason.UNDEFINED_RISK, v.reasons)


class TestSpreadAndLiquidity(unittest.TestCase):
    def test_wide_spread_on_any_leg_vetoes(self):
        legs = list(iron_condor())
        legs[2] = leg(kind="call", side="sell", strike=105.0, bid=0.50, ask=1.50)  # 100% of mid
        v = evaluate(proposal(legs=tuple(legs)), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.WIDE_SPREAD, v.reasons)

    def test_thin_open_interest_vetoes(self):
        legs = list(iron_condor())
        legs[0] = leg(kind="put", side="buy", strike=92.0, oi=10)
        v = evaluate(proposal(legs=tuple(legs)), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.ILLIQUID, v.reasons)

    def test_zero_volume_vetoes(self):
        legs = list(iron_condor())
        legs[1] = leg(kind="put", side="sell", strike=95.0, volume=0)
        v = evaluate(proposal(legs=tuple(legs)), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.ILLIQUID, v.reasons)

    def test_zero_mid_price_is_vetoed_not_divided_by(self):
        legs = list(iron_condor())
        legs[3] = leg(kind="call", side="buy", strike=108.0, bid=0.0, ask=0.0)
        v = evaluate(proposal(legs=tuple(legs)), snapshot())
        self.assertFalse(v.approved)  # must not raise ZeroDivisionError


class TestExpiry(unittest.TestCase):
    def test_expiry_beyond_window_vetoes(self):
        far = iron_condor(expiry=date(2026, 12, 18))  # ~108 DTE
        v = evaluate(proposal(legs=far), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.DTE_OUT_OF_RANGE, v.reasons)

    def test_same_day_expiry_vetoes(self):
        """The live run filled a 0-DTE contract while configured for 30."""
        today = iron_condor(expiry=date(2026, 9, 1))
        v = evaluate(proposal(legs=today), snapshot(now=datetime(2026, 9, 1, 14, 0)))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.DTE_OUT_OF_RANGE, v.reasons)


class TestDeltaNeutrality(unittest.TestCase):
    def test_directional_structure_vetoes(self):
        legs = (leg(kind="call", side="buy", strike=100.0, delta=0.55),
                leg(kind="call", side="sell", strike=110.0, delta=0.10))
        v = evaluate(proposal(legs=legs), snapshot())
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.NOT_DELTA_NEUTRAL, v.reasons)

    def test_balanced_condor_passes_delta_gate(self):
        v = evaluate(proposal(), snapshot())
        self.assertNotIn(VetoReason.NOT_DELTA_NEUTRAL, v.reasons)

    def test_offsetting_legs_net_to_zero_delta(self):
        """Long and short of the SAME delta cancel. This only holds if position
        direction is applied to delta — summing magnitudes would give 0.60 and
        wrongly veto. Guards against dropping `position_sign` from the sum."""
        legs = (leg(kind="call", side="buy",  strike=100.0, delta=0.30),
                leg(kind="call", side="sell", strike=105.0, delta=0.30))
        v = evaluate(proposal(legs=legs), snapshot())
        self.assertAlmostEqual(v.net_delta, 0.0, places=6)
        self.assertNotIn(VetoReason.NOT_DELTA_NEUTRAL, v.reasons)

    def test_two_shorts_are_directional_despite_equal_magnitudes(self):
        """Two short calls are net SHORT delta, not neutral."""
        legs = (leg(kind="call", side="sell", strike=105.0, delta=0.30),
                leg(kind="call", side="buy",  strike=108.0, delta=0.02),
                leg(kind="call", side="sell", strike=110.0, delta=0.30),
                leg(kind="call", side="buy",  strike=112.0, delta=0.02))
        v = evaluate(proposal(legs=legs), snapshot())
        self.assertLess(v.net_delta, -0.5)
        self.assertIn(VetoReason.NOT_DELTA_NEUTRAL, v.reasons)


class TestAccountState(unittest.TestCase):
    def test_daily_loss_kill_switch(self):
        v = evaluate(proposal(), snapshot(day_pnl=-6_500.0))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.DAILY_LOSS_LIMIT, v.reasons)

    def test_drawdown_floor_blocks_convex_sleeve_only(self):
        low = snapshot(equity=80_000.0)
        self.assertFalse(evaluate(proposal(sleeve="convex"), low).approved)
        self.assertTrue(evaluate(proposal(sleeve="core"), low).approved)

    def test_market_closed_vetoes(self):
        v = evaluate(proposal(), snapshot(market_open=False))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.MARKET_CLOSED, v.reasons)

    def test_no_entries_near_the_close(self):
        v = evaluate(proposal(), snapshot(minutes_to_close=8))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.TOO_LATE_IN_SESSION, v.reasons)

    def test_max_concurrent_positions(self):
        many = tuple(Position(underlying=f"SYM{i}", risk=500.0, sleeve="core") for i in range(8))
        v = evaluate(proposal(), snapshot(positions=many))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.TOO_MANY_POSITIONS, v.reasons)

    def test_concentration_cap_per_underlying(self):
        heavy = (Position(underlying="SPY", risk=26_000.0, sleeve="core"),)
        v = evaluate(proposal(underlying="SPY"), snapshot(positions=heavy))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.CONCENTRATED, v.reasons)

    def test_duplicate_submission_is_blocked(self):
        p, s = proposal(), snapshot()
        first = evaluate(p, s)
        self.assertTrue(first.approved)
        again = evaluate(p, snapshot(submitted=frozenset({first.idempotency_key})))
        self.assertFalse(again.approved)
        self.assertIn(VetoReason.DUPLICATE, again.reasons)

    def test_idempotency_key_is_stable_and_specific(self):
        k1 = evaluate(proposal(), snapshot()).idempotency_key
        k2 = evaluate(proposal(), snapshot()).idempotency_key
        k3 = evaluate(proposal(underlying="QQQ"), snapshot()).idempotency_key
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)


class TestSleeveBudget(unittest.TestCase):
    def test_core_and_convex_have_separate_budgets(self):
        cfg = GateConfig()
        self.assertAlmostEqual(cfg.core_fraction + cfg.convex_fraction, 1.0, places=6)

    def test_sleeve_budget_exhausted_vetoes(self):
        # Derived from the configured fraction rather than hardcoded, so the
        # test keeps testing the gate when the sleeve split is retuned.
        budget = 100_000.0 * GateConfig().core_fraction
        full = (Position(underlying="QQQ", risk=budget * 0.9, sleeve="core"),
                Position(underlying="IWM", risk=budget * 0.2, sleeve="core"))
        v = evaluate(proposal(sleeve="core"), snapshot(positions=full))
        self.assertFalse(v.approved)
        self.assertIn(VetoReason.SLEEVE_BUDGET_EXHAUSTED, v.reasons)

    def test_convex_budget_is_independent_of_core(self):
        core_full = (Position(underlying="QQQ",
                              risk=100_000.0 * GateConfig().core_fraction,
                              sleeve="core"),)
        v = evaluate(proposal(sleeve="convex"), snapshot(positions=core_full))
        self.assertNotIn(VetoReason.SLEEVE_BUDGET_EXHAUSTED, v.reasons)


class TestPurity(unittest.TestCase):
    def test_all_vetoes_are_collected_not_just_the_first(self):
        """A judge should see every reason a trade was declined."""
        bad = evaluate(proposal(legs=naked_strangle()),
                       snapshot(market_open=False, day_pnl=-9_000.0))
        self.assertGreaterEqual(len(bad.reasons), 3)

    def test_proposal_is_not_mutated(self):
        p = proposal()
        before = (p.underlying, p.sleeve, p.legs)
        evaluate(p, snapshot())
        self.assertEqual((p.underlying, p.sleeve, p.legs), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
