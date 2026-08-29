"""
Tests for config.StrategyConfig and quant_core.decision_journal.

Two problems these modules address.

Hardcoded 30 DTE. Judging ends 4 Sep. A position opened 30 days out cannot
expire, be assigned, or complete a cycle inside the window, so judges see one
open position and mark-to-market drift instead of a completed round trip. It
also means exit_engine's DTE<=5 liquidation rule never fires in-window.

The dashboard can never read the pipeline log. GitHub Actions writes
pipeline.log onto a runner that is destroyed when the job ends; dashboard.py
reads pipeline.log from its own filesystem. They share no storage, so the
telemetry panel shows "pending generation" forever. A committed JSONL journal
gives both sides one file that actually travels.
"""
import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from config import StrategyConfig
from quant_core.decision_journal import (
    journal_entry, append_entry, read_entries, summarise,
)
from quant_core.risk_gates import (
    Leg, TradeProposal, AccountSnapshot, evaluate,
)


def _leg(kind="call", side="sell", strike=105.0):
    return Leg(symbol=f"SPY260904{'C' if kind == 'call' else 'P'}{int(strike*1000):08d}",
               kind=kind, side=side, strike=strike, ratio=1, delta=0.15,
               bid=0.90, ask=0.94, open_interest=2000, volume=500,
               expiry=date(2026, 9, 4))


def _proposal():
    return TradeProposal(underlying="SPY", structure="iron_condor", sleeve="core",
                         legs=(_leg("put", "buy", 92.0), _leg("put", "sell", 95.0),
                               _leg("call", "sell", 105.0), _leg("call", "buy", 108.0)),
                         rationale="IV rich vs GARCH forecast")


def _snapshot():
    return AccountSnapshot(equity=100_000.0, starting_equity=100_000.0, day_pnl=0.0,
                           now=datetime(2026, 9, 1, 14, 0), market_open=True,
                           minutes_to_close=180)


class TestStrategyConfig(unittest.TestCase):
    def test_default_dte_fits_inside_the_contest_window(self):
        """30 DTE cannot resolve before 4 Sep. Default must be short-dated."""
        cfg = StrategyConfig()
        self.assertLessEqual(cfg.target_dte, 10)
        self.assertGreaterEqual(cfg.target_dte, 1)

    def test_target_dte_sits_within_the_min_max_band(self):
        cfg = StrategyConfig()
        self.assertGreaterEqual(cfg.target_dte, cfg.min_dte)
        self.assertLessEqual(cfg.target_dte, cfg.max_dte)

    def test_env_overrides_defaults(self):
        os.environ["TARGET_DTE"] = "5"
        os.environ["UNDERLYING"] = "QQQ"
        try:
            cfg = StrategyConfig.from_env()
            self.assertEqual(cfg.target_dte, 5)
            self.assertEqual(cfg.underlying, "QQQ")
        finally:
            del os.environ["TARGET_DTE"], os.environ["UNDERLYING"]

    def test_invalid_env_value_falls_back_to_default(self):
        os.environ["TARGET_DTE"] = "not-a-number"
        try:
            self.assertEqual(StrategyConfig.from_env().target_dte,
                             StrategyConfig().target_dte)
        finally:
            del os.environ["TARGET_DTE"]

    def test_inconsistent_band_is_rejected(self):
        with self.assertRaises(ValueError):
            StrategyConfig(min_dte=10, max_dte=2)

    def test_target_outside_band_is_rejected(self):
        with self.assertRaises(ValueError):
            StrategyConfig(min_dte=1, max_dte=10, target_dte=45)


class TestJournalEntry(unittest.TestCase):
    def test_records_an_approval_with_its_reasoning(self):
        p, s = _proposal(), _snapshot()
        e = journal_entry(p, evaluate(p, s), s)
        self.assertTrue(e["approved"])
        self.assertEqual(e["underlying"], "SPY")
        self.assertEqual(e["rationale"], "IV rich vs GARCH forecast")
        self.assertGreaterEqual(e["contracts"], 1)

    def test_records_a_veto_with_every_reason(self):
        """Vetoes are the demo. A judge seeing the agent decline a bad trade
        with stated reasons is worth more than one more fill."""
        p = _proposal()
        s = AccountSnapshot(equity=100_000.0, starting_equity=100_000.0,
                            day_pnl=-9_000.0, now=datetime(2026, 9, 1, 14, 0),
                            market_open=False, minutes_to_close=180)
        e = journal_entry(p, evaluate(p, s), s)
        self.assertFalse(e["approved"])
        self.assertIn("market_closed", e["reasons"])
        self.assertIn("daily_loss_limit", e["reasons"])

    def test_entry_is_json_serialisable(self):
        p, s = _proposal(), _snapshot()
        json.dumps(journal_entry(p, evaluate(p, s), s))  # must not raise

    def test_entry_carries_a_timestamp_and_leg_detail(self):
        p, s = _proposal(), _snapshot()
        e = journal_entry(p, evaluate(p, s), s)
        self.assertIn("timestamp", e)
        self.assertEqual(len(e["legs"]), 4)
        self.assertIn("delta", e["legs"][0])


class TestJournalIO(unittest.TestCase):
    def test_append_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "journal.jsonl"
            p, s = _proposal(), _snapshot()
            append_entry(path, journal_entry(p, evaluate(p, s), s))
            append_entry(path, journal_entry(p, evaluate(p, s), s))
            self.assertEqual(len(read_entries(path)), 2)

    def test_reading_a_missing_file_returns_empty_not_an_error(self):
        """The dashboard must render on a fresh deploy with no journal yet."""
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_entries(Path(d) / "nope.jsonl"), [])

    def test_a_corrupt_line_does_not_destroy_the_whole_journal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "journal.jsonl"
            p, s = _proposal(), _snapshot()
            append_entry(path, journal_entry(p, evaluate(p, s), s))
            with open(path, "a") as f:
                f.write("{ this is not json\n")
            append_entry(path, journal_entry(p, evaluate(p, s), s))
            self.assertEqual(len(read_entries(path)), 2)  # skips the bad line

    def test_summarise_counts_approvals_and_vetoes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "journal.jsonl"
            p, good = _proposal(), _snapshot()
            bad = AccountSnapshot(equity=100_000.0, starting_equity=100_000.0,
                                  day_pnl=0.0, now=datetime(2026, 9, 1, 14, 0),
                                  market_open=False, minutes_to_close=180)
            append_entry(path, journal_entry(p, evaluate(p, good), good))
            append_entry(path, journal_entry(p, evaluate(p, bad), bad))
            append_entry(path, journal_entry(p, evaluate(p, bad), bad))
            s = summarise(read_entries(path))
            self.assertEqual(s["total"], 3)
            self.assertEqual(s["approved"], 1)
            self.assertEqual(s["vetoed"], 2)
            self.assertEqual(s["top_veto_reasons"][0][0], "market_closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
