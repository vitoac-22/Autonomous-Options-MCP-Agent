"""
A durable record of what the agent decided and why.

The dashboard cannot currently see anything the pipeline did. GitHub Actions
writes `pipeline.log` onto a runner that is destroyed when the job ends, while
`dashboard.py` reads `pipeline.log` from its own filesystem. The two share no
storage, so the telemetry panel shows "Audit file pending generation" forever in
production. It is an architecture gap, not a bug in either file.

A JSONL journal committed back to the repository after each run gives both sides
one file that actually travels: the workflow appends and commits, the dashboard
reads it from the checkout it was deployed from. No database to provision, no
extra secrets — which matters more than elegance with a week on the clock.

Vetoes are recorded as carefully as fills. An agent that declines a bad trade
and can say exactly why is a better demo than one that simply trades more.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_JOURNAL = Path("data/decisions.jsonl")


def journal_entry(proposal, verdict, snapshot) -> dict[str, Any]:
    """Build one journal record. Pure — no I/O, no clock read.

    The timestamp comes from the snapshot, so a replayed run reproduces exactly
    the record it produced the first time.
    """
    return {
        "timestamp": snapshot.now.isoformat(),
        "underlying": proposal.underlying,
        "structure": proposal.structure,
        "sleeve": proposal.sleeve,
        "rationale": proposal.rationale,
        "approved": verdict.approved,
        "contracts": verdict.contracts,
        "reasons": list(verdict.reasons),
        "max_loss_per_contract": verdict.max_loss_per_contract,
        "net_delta": verdict.net_delta,
        "idempotency_key": verdict.idempotency_key,
        "equity": snapshot.equity,
        "day_pnl": snapshot.day_pnl,
        "legs": [
            {
                "symbol": leg.symbol,
                "kind": leg.kind,
                "side": leg.side,
                "strike": leg.strike,
                "ratio": leg.ratio,
                "delta": leg.delta,
                "mid": leg.mid,
                "open_interest": leg.open_interest,
                "expiry": leg.expiry.isoformat(),
            }
            for leg in proposal.legs
        ],
    }


def append_entry(path: Path | str, entry: dict[str, Any]) -> None:
    """Append one record. Creates the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def read_entries(path: Path | str) -> list[dict[str, Any]]:
    """Read every valid record.

    A missing file yields an empty list rather than raising, so the dashboard
    renders on a fresh deploy. A malformed line is skipped rather than
    discarding the whole journal — one truncated write should not erase the
    week's history.
    """
    path = Path(path)
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarise(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Headline counts for the dashboard."""
    entries = list(entries)
    approved = [e for e in entries if e.get("approved")]
    vetoed = [e for e in entries if not e.get("approved")]

    reasons: Counter = Counter()
    for entry in vetoed:
        reasons.update(entry.get("reasons", []))

    return {
        "total": len(entries),
        "approved": len(approved),
        "vetoed": len(vetoed),
        "contracts_traded": sum(e.get("contracts", 0) for e in approved),
        "top_veto_reasons": reasons.most_common(5),
        "last_decision": entries[-1]["timestamp"] if entries else None,
    }
