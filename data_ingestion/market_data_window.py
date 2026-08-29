"""
Market-data window helper for Alpaca's free Basic plan.

The free plan refuses historical bars inside the last 15 minutes:

    APIError: subscription does not permit querying recent SIP data

The pipeline asked for bars through `end_date = today`, which fails outright
during market hours — so the scheduled run would have died before GARCH ever
started. Quotes and option chains are real-time on the free plan; only the
historical bar endpoint is delayed, which is why this only affects the
underlying ingest.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# The documented limit is 15 minutes. A little margin absorbs clock skew
# between this machine and Alpaca.
SIP_DELAY_MINUTES = 20


def safe_end_timestamp(now: datetime | None = None) -> datetime:
    """The latest bar timestamp the free plan will serve."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now - timedelta(minutes=SIP_DELAY_MINUTES)
