"""
Tests for data_ingestion.options_market_data.

The pipeline currently invents its inputs:

    d_proxy = 0.1 if abs(leg['sign']) == 1 else 0.5   # sign is always +/-1 -> always 0.1
    precios_estimados.append(2.5)                     # every leg, every time

So the CVXPY optimiser minimises net delta across four legs that all carry the
same made-up delta, under a capital constraint built from made-up prices. This
module replaces both with real values from Alpaca's option snapshots.

The network call and the mapping are deliberately separate: `build_leg` is pure
and gets tested exhaustively here, while the client wrapper is a thin shell over
the SDK.
"""
import unittest
from datetime import date

from data_ingestion.options_market_data import (
    build_leg, parse_occ_symbol, MissingGreeksError,
)


class FakeGreeks:
    def __init__(self, delta): self.delta = delta


class FakeQuote:
    def __init__(self, bid, ask): self.bid_price, self.ask_price = bid, ask


class FakeSnapshot:
    """Mirrors the shape alpaca-py returns from get_option_snapshot."""
    def __init__(self, delta=0.16, bid=0.90, ask=0.94, iv=0.21,
                 oi=2500, volume=800, greeks=True):
        self.greeks = FakeGreeks(delta) if greeks else None
        self.latest_quote = FakeQuote(bid, ask)
        self.implied_volatility = iv
        self.open_interest = oi
        self.volume = volume


class TestOCCParsing(unittest.TestCase):
    def test_parses_standard_occ_symbol(self):
        got = parse_occ_symbol("SPY260904C00105000")
        self.assertEqual(got.underlying, "SPY")
        self.assertEqual(got.expiry, date(2026, 9, 4))
        self.assertEqual(got.kind, "call")
        self.assertAlmostEqual(got.strike, 105.0)

    def test_parses_put_and_fractional_strike(self):
        got = parse_occ_symbol("SPY260904P00092500")
        self.assertEqual(got.kind, "put")
        self.assertAlmostEqual(got.strike, 92.5)

    def test_handles_four_character_underlying(self):
        """exit_engine slices symbol[3:9], which breaks on non-3-char tickers."""
        got = parse_occ_symbol("QQQQ260904C00105000")
        self.assertEqual(got.underlying, "QQQQ")
        self.assertEqual(got.expiry, date(2026, 9, 4))
        self.assertAlmostEqual(got.strike, 105.0)

    def test_rejects_malformed_symbol(self):
        with self.assertRaises(ValueError):
            parse_occ_symbol("NOTASYMBOL")


class TestBuildLeg(unittest.TestCase):
    def test_uses_real_delta_not_a_proxy(self):
        leg = build_leg("SPY260904C00105000", "sell", FakeSnapshot(delta=0.1637), ratio=1)
        self.assertAlmostEqual(leg.delta, 0.1637)
        self.assertNotAlmostEqual(leg.delta, 0.1, places=3)  # the old hardcoded value

    def test_uses_real_mid_price_not_2_50(self):
        leg = build_leg("SPY260904C00105000", "sell", FakeSnapshot(bid=1.20, ask=1.30), ratio=1)
        self.assertAlmostEqual(leg.mid, 1.25)
        self.assertNotAlmostEqual(leg.mid, 2.5, places=3)  # the old hardcoded value

    def test_carries_liquidity_fields_for_the_gates(self):
        leg = build_leg("SPY260904C00105000", "sell",
                        FakeSnapshot(oi=1234, volume=567), ratio=1)
        self.assertEqual(leg.open_interest, 1234)
        self.assertEqual(leg.volume, 567)

    def test_derives_kind_strike_and_expiry_from_the_symbol(self):
        leg = build_leg("SPY260904P00092500", "buy", FakeSnapshot(), ratio=1)
        self.assertEqual(leg.kind, "put")
        self.assertAlmostEqual(leg.strike, 92.5)
        self.assertEqual(leg.expiry, date(2026, 9, 4))

    def test_position_sign_follows_side(self):
        self.assertEqual(build_leg("SPY260904C00105000", "buy", FakeSnapshot(), 1).position_sign, 1)
        self.assertEqual(build_leg("SPY260904C00105000", "sell", FakeSnapshot(), 1).position_sign, -1)

    def test_missing_greeks_raises_rather_than_defaulting(self):
        """Silently substituting a default delta is exactly the bug being fixed."""
        with self.assertRaises(MissingGreeksError):
            build_leg("SPY260904C00105000", "sell", FakeSnapshot(greeks=False), ratio=1)

    def test_missing_quote_raises(self):
        snap = FakeSnapshot()
        snap.latest_quote = None
        with self.assertRaises(MissingGreeksError):
            build_leg("SPY260904C00105000", "sell", snap, ratio=1)

    def test_none_open_interest_becomes_zero_so_gates_veto(self):
        """Absent liquidity data must read as illiquid, never as unlimited."""
        snap = FakeSnapshot()
        snap.open_interest, snap.volume = None, None
        leg = build_leg("SPY260904C00105000", "sell", snap, ratio=1)
        self.assertEqual(leg.open_interest, 0)
        self.assertEqual(leg.volume, 0)

    def test_result_feeds_the_gates_directly(self):
        """build_leg must return the exact Leg type risk_gates consumes."""
        from quant_core.risk_gates import Leg
        leg = build_leg("SPY260904C00105000", "sell", FakeSnapshot(), ratio=1)
        self.assertIsInstance(leg, Leg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
