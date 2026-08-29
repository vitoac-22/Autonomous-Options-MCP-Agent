"""
Tests for ai_agent.mleg_payload.

Two defects in the current builder:

1. The payload omits `qty` (top level) and `position_intent` (per leg), both of
   which Alpaca requires for order_class="mleg". The order is rejected 422, so
   the atomic execution path has never actually placed an order.

2. ratio_qty = int(leg['qty'] / base_qty) truncates. [4,6,6,4] collapses to
   [1,1,1,1], silently turning a 2:3:3:2 structure into an equal-ratio one —
   a different risk profile than the optimiser computed.
"""
import unittest
from datetime import date
from math import gcd
from functools import reduce

from ai_agent.mleg_payload import build_mleg_payload, PositionIntent
from quant_core.risk_gates import Leg


def leg(kind="call", side="sell", strike=105.0, ratio=1, bid=None, ask=None):
    # Wings cheap, bodies rich, so a condor nets a real credit rather than zero.
    if bid is None:
        bid = 0.40 if strike in (92.0, 108.0) else 0.90
    if ask is None:
        ask = bid + 0.04
    return Leg(symbol=f"SPY260904{'C' if kind == 'call' else 'P'}{int(strike*1000):08d}",
               kind=kind, side=side, strike=strike, ratio=ratio, delta=0.15,
               bid=bid, ask=ask, open_interest=2000, volume=500,
               expiry=date(2026, 9, 4))


def condor(ratios=(1, 1, 1, 1)):
    return (leg("put", "buy", 92.0, ratios[0]),
            leg("put", "sell", 95.0, ratios[1]),
            leg("call", "sell", 105.0, ratios[2]),
            leg("call", "buy", 108.0, ratios[3]))


class TestRequiredFields(unittest.TestCase):
    def test_top_level_qty_is_present(self):
        """Missing today. Alpaca requires it for mleg; without it: 422."""
        p = build_mleg_payload(condor(), contracts=3)
        self.assertIn("qty", p)
        self.assertEqual(p["qty"], "3")

    def test_every_leg_has_position_intent(self):
        """Missing today. Required per leg."""
        p = build_mleg_payload(condor(), contracts=1)
        for l in p["legs"]:
            self.assertIn("position_intent", l)

    def test_opening_intents_match_side(self):
        p = build_mleg_payload(condor(), contracts=1)
        intents = [l["position_intent"] for l in p["legs"]]
        self.assertEqual(intents, ["buy_to_open", "sell_to_open",
                                   "sell_to_open", "buy_to_open"])

    def test_closing_intents_match_side(self):
        p = build_mleg_payload(condor(), contracts=1, intent=PositionIntent.CLOSE)
        intents = [l["position_intent"] for l in p["legs"]]
        self.assertEqual(intents, ["buy_to_close", "sell_to_close",
                                   "sell_to_close", "buy_to_close"])

    def test_order_class_is_mleg(self):
        self.assertEqual(build_mleg_payload(condor(), contracts=1)["order_class"], "mleg")

    def test_all_numeric_fields_are_strings(self):
        """The current system prompt begs the model to send strings. Cast in
        code instead of asking a model to remember."""
        p = build_mleg_payload(condor(), contracts=2, limit_price=1.25)
        self.assertIsInstance(p["qty"], str)
        self.assertIsInstance(p["limit_price"], str)
        for l in p["legs"]:
            self.assertIsInstance(l["ratio_qty"], str)


class TestRatioTruncation(unittest.TestCase):
    def test_the_truncation_bug_case(self):
        """[4,6,6,4] must become 2:3:3:2. The old int(q/min) gave [1,1,1,1]."""
        p = build_mleg_payload(condor((4, 6, 6, 4)), contracts=1)
        self.assertEqual([l["ratio_qty"] for l in p["legs"]], ["2", "3", "3", "2"])

    def test_second_truncation_case(self):
        """[3,5,5,3] has gcd 1 — ratios pass through unchanged."""
        p = build_mleg_payload(condor((3, 5, 5, 3)), contracts=1)
        self.assertEqual([l["ratio_qty"] for l in p["legs"]], ["3", "5", "5", "3"])

    def test_ratios_are_always_reduced_to_gcd_one(self):
        """Alpaca requires the ratio set be in lowest terms."""
        for ratios in [(1, 1, 1, 1), (2, 4, 4, 2), (4, 6, 6, 4),
                       (3, 5, 5, 3), (10, 20, 20, 10), (6, 9, 9, 6)]:
            p = build_mleg_payload(condor(ratios), contracts=1)
            got = [int(l["ratio_qty"]) for l in p["legs"]]
            self.assertEqual(reduce(gcd, got), 1, msg=f"{ratios} -> {got}")

    def test_reduction_preserves_proportions(self):
        p = build_mleg_payload(condor((10, 20, 20, 10)), contracts=1)
        got = [int(l["ratio_qty"]) for l in p["legs"]]
        self.assertEqual(got, [1, 2, 2, 1])

    def test_no_ratio_is_ever_zero(self):
        for ratios in [(1, 7, 7, 1), (1, 2, 3, 4), (5, 1, 1, 5)]:
            p = build_mleg_payload(condor(ratios), contracts=1)
            for l in p["legs"]:
                self.assertGreaterEqual(int(l["ratio_qty"]), 1)


class TestOrderType(unittest.TestCase):
    def test_limit_order_when_price_given(self):
        p = build_mleg_payload(condor(), contracts=1, limit_price=2.50)
        self.assertEqual(p["type"], "limit")
        self.assertEqual(p["limit_price"], "2.5")

    def test_market_order_when_no_price(self):
        p = build_mleg_payload(condor(), contracts=1)
        self.assertEqual(p["type"], "market")
        self.assertNotIn("limit_price", p)

    def test_time_in_force_is_day(self):
        """Options only accept day or gtc."""
        self.assertEqual(build_mleg_payload(condor(), contracts=1)["time_in_force"], "day")


class TestValidation(unittest.TestCase):
    def test_zero_contracts_rejected(self):
        with self.assertRaises(ValueError):
            build_mleg_payload(condor(), contracts=0)

    def test_no_legs_rejected(self):
        with self.assertRaises(ValueError):
            build_mleg_payload((), contracts=1)

    def test_too_many_legs_rejected(self):
        with self.assertRaises(ValueError):
            build_mleg_payload(condor() + condor(), contracts=1)  # 8 legs

    def test_zero_limit_price_rejected(self):
        """A $0 limit accepts no credit and can never fill. Refuse it."""
        with self.assertRaises(ValueError):
            build_mleg_payload(condor(), contracts=1, limit_price=0.0)

    def test_negative_limit_price_rejected(self):
        with self.assertRaises(ValueError):
            build_mleg_payload(condor(), contracts=1, limit_price=-1.5)

    def test_net_limit_price_is_positive_for_a_credit_condor(self):
        from ai_agent.mleg_payload import net_limit_price
        self.assertGreater(net_limit_price(condor()), 0)

    def test_leg_symbols_are_preserved_exactly(self):
        legs = condor()
        p = build_mleg_payload(legs, contracts=1)
        self.assertEqual([l["symbol"] for l in p["legs"]],
                         [l.symbol for l in legs])


if __name__ == "__main__":
    unittest.main(verbosity=2)
