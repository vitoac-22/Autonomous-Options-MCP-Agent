"""
Option chain resolution: choose one expiry, then map every strike inside it.

Replaces OptionsContractResolver.get_valid_occ_symbol, which has four defects
that compound into a structure that is not the one the optimiser sized.

Put/call filtering was `if target_char not in symbol` — a substring test over
the whole OCC symbol. "SPY" contains a P, so the put filter is a no-op for this
underlying and admits calls. Ties on (distance, dte) break by response order and
"C" precedes "P", so a put leg reliably took a call. Alpaca accepts a
server-side `type=ContractType.PUT` filter; it was imported nowhere and never
used. Filtering now happens on the server, and again locally on the parsed type.

The request set `expiration_date_gte` with no `expiration_date_lte`, no `limit`,
and never read `next_page_token` — asking for every expiry from +3 days out to
the furthest LEAPS and seeing 100 arbitrary rows of it. The window is now
bounded on both sides and every page is fetched.

Most importantly, each leg resolved independently and strike distance outranked
DTE in the sort, so nothing pinned the four legs to a common expiration. An iron
condor could land across four different expiry dates, at which point it is not a
condor and the defined-risk structure is gone. Expiry is now chosen once, up
front, and every strike is mapped within it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence


class NoExpiryInWindow(RuntimeError):
    """No expiration in the chain falls inside the configured DTE band."""


class StrikeNotAvailable(RuntimeError):
    """The chosen expiry carries no contract of the type a leg requires."""


@dataclass(frozen=True)
class Contract:
    symbol: str
    underlying: str
    expiry: date
    kind: str            # "call" | "put"
    strike: float
    open_interest: int = 0


@dataclass(frozen=True)
class LegSpec:
    kind: str            # "call" | "put"
    side: str            # "buy" | "sell"
    target_strike: float
    ratio: int = 1


# ── pure selection logic ────────────────────────────────────────────────────
def select_expiry(contracts: Iterable[Contract],
                  today: date,
                  target_dte: int,
                  min_dte: int,
                  max_dte: int) -> date:
    """Pick one expiration for the whole structure.

    Chooses the expiry closest to `target_dte` among those inside the band that
    carry both calls and puts — an expiry missing either cannot host a condor.
    Pure: `today` is passed in rather than read from the clock.
    """
    by_expiry: dict[date, set[str]] = {}
    for contract in contracts:
        by_expiry.setdefault(contract.expiry, set()).add(contract.kind.lower())

    eligible = [
        expiry for expiry, kinds in by_expiry.items()
        if min_dte <= (expiry - today).days <= max_dte
        and {"call", "put"} <= kinds
    ]
    if not eligible:
        raise NoExpiryInWindow(
            f"No expiry with both calls and puts within {min_dte}-{max_dte} DTE "
            f"of {today}. Saw: {sorted(by_expiry)}"
        )

    return min(eligible, key=lambda e: (abs((e - today).days - target_dte), e))


def map_legs_to_contracts(contracts: Iterable[Contract],
                          expiry: date,
                          legs: Sequence[LegSpec]) -> list[Contract]:
    """Snap each leg to the nearest available strike **within `expiry`**.

    Contracts on other expiries are discarded before matching, so a tempting
    exact strike on the wrong date can never pull a leg out of the structure.
    Type is matched on the parsed `kind`, never on a substring of the symbol.
    """
    pool = [c for c in contracts if c.expiry == expiry]

    resolved: list[Contract] = []
    used: set[str] = set()

    for leg in legs:
        kind = leg.kind.lower()
        candidates = [c for c in pool
                      if c.kind.lower() == kind and c.symbol not in used]
        if not candidates:
            raise StrikeNotAvailable(
                f"No unused {kind} contracts at {expiry} for target strike "
                f"{leg.target_strike}."
            )
        best = min(candidates, key=lambda c: (abs(c.strike - leg.target_strike), c.strike))
        used.add(best.symbol)
        resolved.append(best)

    return resolved


# ── the I/O shell ───────────────────────────────────────────────────────────
class ChainResolver:
    """Fetches the option chain from Alpaca and resolves a structure inside it."""

    def __init__(self, underlying: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        from alpaca.trading.client import TradingClient

        self.underlying = underlying.upper()
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        api_secret = (api_secret or os.getenv("ALPACA_SECRET_KEY")
                      or os.getenv("ALPACA_API_SECRET"))
        if not api_key or not api_secret:
            raise ValueError("Missing Alpaca credentials in environment.")
        self.trading_client = TradingClient(api_key, api_secret, paper=True)

    def _fetch_all_pages(self, request, max_pages: int = 20) -> list:
        """Follow `next_page_token` to the end.

        The previous resolver read page one and dropped the token, so it saw at
        most 100 contracts of a chain spanning every expiry. `max_pages` bounds
        the loop so a server that always returns a token cannot hang the run.
        """
        collected: list = []
        for _ in range(max_pages):
            response = self.trading_client.get_option_contracts(request)
            collected.extend(response.option_contracts or [])
            token = getattr(response, "next_page_token", None)
            if not token:
                break
            try:
                request.page_token = token
            except Exception as exc:
                # Returning page one here is precisely the silent truncation
                # this function exists to remove, so fail loudly instead.
                raise RuntimeError(
                    "Could not advance the option-chain page token; refusing to "
                    "continue with a partial chain."
                ) from exc
        return collected

    def fetch_chain(self, today: date, min_dte: int, max_dte: int,
                    strike_low: float, strike_high: float) -> list[Contract]:
        """Both-sided expiry window, both option types, every page."""
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType

        out: list[Contract] = []
        for kind, enum in (("call", ContractType.CALL), ("put", ContractType.PUT)):
            request = GetOptionContractsRequest(
                underlying_symbols=[self.underlying],
                status="active",
                type=enum,                       # server-side; never used before
                expiration_date_gte=(today + timedelta(days=min_dte)).strftime("%Y-%m-%d"),
                expiration_date_lte=(today + timedelta(days=max_dte)).strftime("%Y-%m-%d"),
                strike_price_gte=str(round(strike_low, 2)),
                strike_price_lte=str(round(strike_high, 2)),
                limit=10_000,
            )
            for raw in self._fetch_all_pages(request):
                parsed = self._to_contract(raw, kind)
                if parsed is not None:
                    out.append(parsed)
        return out

    @staticmethod
    def _to_contract(raw, fallback_kind: str) -> Optional[Contract]:
        symbol = str(getattr(raw, "symbol", "") or "")
        if len(symbol) < 15:
            return None

        # Trust the parsed symbol over the request we think we made.
        kind = "call" if symbol[-9] == "C" else "put" if symbol[-9] == "P" else fallback_kind

        strike = getattr(raw, "strike_price", None)
        try:
            strike = float(strike) if strike is not None else int(symbol[-8:]) / 1000.0
        except (TypeError, ValueError):
            return None

        raw_expiry = getattr(raw, "expiration_date", None)
        try:
            if isinstance(raw_expiry, date):
                expiry = raw_expiry
            elif raw_expiry:
                expiry = datetime.strptime(str(raw_expiry)[:10], "%Y-%m-%d").date()
            else:
                ymd = symbol[-15:-9]
                expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        except (TypeError, ValueError):
            return None

        return Contract(
            symbol=symbol,
            underlying=symbol[:-15],
            expiry=expiry,
            kind=kind,
            strike=strike,
            open_interest=int(getattr(raw, "open_interest", None) or 0),
        )

    def resolve_structure(self, legs: Sequence[LegSpec], today: date,
                          target_dte: int, min_dte: int, max_dte: int,
                          strike_pad: float = 20.0) -> tuple[date, list[Contract]]:
        """Choose one expiry, then map every leg inside it."""
        strikes = [leg.target_strike for leg in legs]
        chain = self.fetch_chain(
            today=today, min_dte=min_dte, max_dte=max_dte,
            strike_low=min(strikes) - strike_pad,
            strike_high=max(strikes) + strike_pad,
        )
        expiry = select_expiry(chain, today, target_dte, min_dte, max_dte)
        return expiry, map_legs_to_contracts(chain, expiry, legs)
