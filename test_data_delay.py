"""
Tests for the free-plan market-data delay guard.

Alpaca's free Basic plan refuses historical bars inside the last 15 minutes:
    APIError: subscription does not permit querying recent SIP data

The pipeline requested bars through `end_date = today`, which fails outright
during market hours — so the Monday run would have died before GARCH ever ran.
"""
import unittest
from datetime import datetime, timedelta, timezone

from data_ingestion.market_data_window import safe_end_timestamp, SIP_DELAY_MINUTES


class TestSafeEndTimestamp(unittest.TestCase):
    def test_end_is_pushed_back_behind_the_sip_delay(self):
        now = datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc)
        end = safe_end_timestamp(now)
        self.assertLessEqual(end, now - timedelta(minutes=SIP_DELAY_MINUTES))

    def test_delay_has_margin_over_the_documented_fifteen_minutes(self):
        self.assertGreater(SIP_DELAY_MINUTES, 15)

    def test_result_is_timezone_aware_utc(self):
        end = safe_end_timestamp(datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc))
        self.assertIsNotNone(end.tzinfo)

    def test_naive_input_is_treated_as_utc_rather_than_crashing(self):
        end = safe_end_timestamp(datetime(2026, 8, 31, 15, 0))
        self.assertIsNotNone(end.tzinfo)

    def test_uses_the_clock_when_no_time_is_given(self):
        before = datetime.now(timezone.utc)
        end = safe_end_timestamp()
        self.assertLessEqual(end, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
