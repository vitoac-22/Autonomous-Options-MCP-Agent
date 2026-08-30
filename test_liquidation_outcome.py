"""
Tests for the liquidation outcome check.

CI run 33322092186 reported success while this happened:

    [ERROR] Failed to liquidate SPY260831P00675000:
            "options market orders are only allowed during market hours"
    [ERROR] LIQUIDATION INCOMPLETE — still exposed: ['SPY260831P00675000']
    [INFO]  === MONITORING CYCLE COMPLETE ===          <- exit 0, CI green

liquidate_portfolio returns the symbols it could not close, and the pipeline
discarded that return value. An emergency exit failed and nothing said so —
the same failure-reported-as-success shape as the swallowed exception and the
discarded broker rejection before it.

The distinction that matters: outside market hours a failed liquidation is not
an emergency, it is "cannot act yet", and failing the build every weekend would
train everyone to ignore a red icon. During market hours it is a real alarm.
"""
import unittest

from quant_core.liquidation_outcome import assess_liquidation, LiquidationOutcome


class TestCleanLiquidation(unittest.TestCase):
    def test_no_failures_is_success(self):
        out = assess_liquidation(failures=[], market_open=True)
        self.assertTrue(out.ok)
        self.assertEqual(out.exit_code, 0)

    def test_no_failures_while_closed_is_also_success(self):
        self.assertTrue(assess_liquidation(failures=[], market_open=False).ok)


class TestFailureDuringMarketHours(unittest.TestCase):
    """Positions the exit engine wanted closed are still open, and we could
    have closed them. That is an alarm."""

    def test_is_not_ok(self):
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=True)
        self.assertFalse(out.ok)

    def test_exits_non_zero_so_ci_goes_red(self):
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=True)
        self.assertEqual(out.exit_code, 1)

    def test_reason_names_the_stranded_symbols(self):
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=True)
        self.assertIn("SPY260831P00675000", out.reason)


class TestFailureOutsideMarketHours(unittest.TestCase):
    """Alpaca refuses option market orders when the market is shut. Nothing is
    broken; we simply cannot act until the open."""

    def test_does_not_fail_the_build(self):
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=False)
        self.assertEqual(out.exit_code, 0)

    def test_is_still_flagged_as_not_ok(self):
        """Exit 0 for CI, but the run must not claim a clean portfolio."""
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=False)
        self.assertFalse(out.ok)

    def test_reason_explains_the_deferral_rather_than_alarming(self):
        out = assess_liquidation(failures=["SPY260831P00675000"], market_open=False)
        self.assertIn("market", out.reason.lower())
        self.assertIn("open", out.reason.lower())


class TestMultipleFailures(unittest.TestCase):
    def test_all_symbols_are_reported(self):
        out = assess_liquidation(failures=["A", "B", "C"], market_open=True)
        for s in ("A", "B", "C"):
            self.assertIn(s, out.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
