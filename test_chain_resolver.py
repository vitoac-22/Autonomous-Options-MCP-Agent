"""
Tests for data_ingestion.chain_resolver.

The existing OptionsContractResolver has four defects that compound:

1. Put/call filtering is `if target_char not in symbol` — a substring test over
   the whole OCC symbol. "SPY" contains a P, so the put filter is a no-op for
   this underlying and admits calls. Sorting ties on (distance, dte) break by
   response order, and "C" precedes "P", so a put leg reliably takes a call.

2. `GetOptionContractsRequest` accepts a server-side `type=ContractType.PUT`
   filter. It is imported nowhere and never used.

3. The request sets `expiration_date_gte` but no `expiration_date_lte`, no
   `limit`, and never reads `next_page_token` — so it asks for every expiry
   from +3 days to the furthest LEAPS and sees 100 arbitrary rows of it.

4. Each leg resolves independently, and strike distance outranks DTE in the
   sort. Nothing pins the four legs to a common expiration, so an iron condor
   can end up spread across four different expiry dates — at which point it is
   not a condor and the defined-risk structure is gone.

The fix is structural: choose ONE expiry first, then map every strike inside it.
"""
import unittest
from datetime import date
from types import SimpleNamespace

from data_ingestion.chain_resolver import (
    Contract, LegSpec, select_expiry, map_legs_to_contracts,
    NoExpiryInWindow, StrikeNotAvailable,
)


def c(strike, kind, expiry=date(2026, 9, 4), oi=1000):
    root = "SPY"
    cp = "C" if kind == "call" else "P"
    return Contract(
        symbol=f"{root}{expiry:%y%m%d}{cp}{int(strike * 1000):08d}",
        underlying=root, expiry=expiry, kind=kind, strike=float(strike),
        open_interest=oi,
    )


def chain(expiries=(date(2026, 9, 4),), strikes=(92, 95, 100, 105, 108)):
    out = []
    for e in expiries:
        for s in strikes:
            out.append(c(s, "put", e))
            out.append(c(s, "call", e))
    return out


CONDOR = (
    LegSpec(kind="put",  side="buy",  target_strike=92.0),
    LegSpec(kind="put",  side="sell", target_strike=95.0),
    LegSpec(kind="call", side="sell", target_strike=105.0),
    LegSpec(kind="call", side="buy",  target_strike=108.0),
)

TODAY = date(2026, 9, 1)


class TestSelectExpiry(unittest.TestCase):
    def test_picks_the_expiry_closest_to_target_dte(self):
        expiries = [date(2026, 9, 2), date(2026, 9, 8), date(2026, 9, 30)]
        got = select_expiry(chain(expiries), today=TODAY, target_dte=7,
                            min_dte=1, max_dte=10)
        self.assertEqual(got, date(2026, 9, 8))   # 7 DTE exactly

    def test_ignores_expiries_outside_the_band(self):
        expiries = [date(2026, 9, 2), date(2026, 12, 18)]
        got = select_expiry(chain(expiries), today=TODAY, target_dte=7,
                            min_dte=1, max_dte=10)
        self.assertEqual(got, date(2026, 9, 2))   # the LEAPS is out of band

    def test_raises_when_nothing_is_in_the_band(self):
        with self.assertRaises(NoExpiryInWindow):
            select_expiry(chain([date(2026, 12, 18)]), today=TODAY,
                          target_dte=7, min_dte=1, max_dte=10)

    def test_same_day_expiry_is_excluded_by_min_dte(self):
        """The live run once filled a 0-DTE contract while aiming for 30."""
        with self.assertRaises(NoExpiryInWindow):
            select_expiry(chain([TODAY]), today=TODAY, target_dte=7,
                          min_dte=1, max_dte=10)

    def test_requires_the_expiry_to_carry_both_calls_and_puts(self):
        """An expiry with only calls cannot host a condor."""
        puts_only = [c(95, "put", date(2026, 9, 2))]
        full = chain([date(2026, 9, 8)])
        got = select_expiry(puts_only + full, today=TODAY, target_dte=1,
                            min_dte=1, max_dte=10)
        self.assertEqual(got, date(2026, 9, 8))


class TestPutCallCorrectness(unittest.TestCase):
    def test_put_legs_resolve_to_puts_not_calls(self):
        """The SPY-contains-P bug. Both types exist at every strike here, so a
        substring filter would hand a call to the put leg."""
        got = map_legs_to_contracts(chain(), date(2026, 9, 4), CONDOR)
        for leg, contract in zip(CONDOR, got):
            self.assertEqual(contract.kind, leg.kind,
                             msg=f"{leg.kind} leg resolved to {contract.kind}: {contract.symbol}")

    def test_resolved_symbols_have_the_right_type_character(self):
        got = map_legs_to_contracts(chain(), date(2026, 9, 4), CONDOR)
        for leg, contract in zip(CONDOR, got):
            self.assertEqual(contract.symbol[-9], "P" if leg.kind == "put" else "C")

    def test_a_condor_resolves_to_exactly_two_puts_and_two_calls(self):
        got = map_legs_to_contracts(chain(), date(2026, 9, 4), CONDOR)
        kinds = [x.kind for x in got]
        self.assertEqual(kinds.count("put"), 2)
        self.assertEqual(kinds.count("call"), 2)


class TestSingleExpiry(unittest.TestCase):
    def test_all_legs_land_on_the_chosen_expiry(self):
        """The structural fix: one expiry chosen first, all strikes mapped in it."""
        multi = chain([date(2026, 9, 2), date(2026, 9, 4), date(2026, 9, 30)])
        got = map_legs_to_contracts(multi, date(2026, 9, 4), CONDOR)
        self.assertEqual({x.expiry for x in got}, {date(2026, 9, 4)})

    def test_legs_cannot_split_across_expiries_even_when_a_closer_strike_exists(self):
        """A tempting exact strike on the WRONG expiry must not win."""
        chosen = chain([date(2026, 9, 4)], strikes=(90, 95, 105, 110))
        tempting = [c(92, "put", date(2026, 9, 30)),      # exact match, wrong expiry
                    c(108, "call", date(2026, 9, 30))]
        got = map_legs_to_contracts(chosen + tempting, date(2026, 9, 4), CONDOR)
        self.assertEqual({x.expiry for x in got}, {date(2026, 9, 4)})


class TestStrikeSnapping(unittest.TestCase):
    def test_snaps_to_the_nearest_available_strike(self):
        got = map_legs_to_contracts(chain(strikes=(90, 96, 104, 110)),
                                    date(2026, 9, 4), CONDOR)
        self.assertEqual([x.strike for x in got], [90.0, 96.0, 104.0, 110.0])

    def test_never_returns_the_same_contract_twice(self):
        legs = (LegSpec("put", "buy", 95.0), LegSpec("put", "sell", 95.2))
        got = map_legs_to_contracts(chain(strikes=(95, 96)), date(2026, 9, 4), legs)
        self.assertEqual(len({x.symbol for x in got}), 2)

    def test_preserves_wing_ordering_for_a_condor(self):
        """long put < short put < short call < long call — or it is not a condor."""
        got = map_legs_to_contracts(chain(), date(2026, 9, 4), CONDOR)
        s = [x.strike for x in got]
        self.assertLess(s[0], s[1])
        self.assertLess(s[1], s[2])
        self.assertLess(s[2], s[3])

    def test_raises_when_a_leg_has_no_contract_of_its_type(self):
        calls_only = [c(s, "call", date(2026, 9, 4)) for s in (100, 105, 110)]
        with self.assertRaises(StrikeNotAvailable):
            map_legs_to_contracts(calls_only, date(2026, 9, 4), CONDOR)


class TestPagination(unittest.TestCase):
    def test_fetches_every_page(self):
        """The old resolver read page one and dropped next_page_token."""
        from data_ingestion.chain_resolver import ChainResolver

        pages = [
            (["a", "b"], "tok1"),
            (["c", "d"], "tok2"),
            (["e"], None),
        ]
        calls = []

        class FakeResponse:
            def __init__(self, items, token):
                self.option_contracts, self.next_page_token = items, token

        class FakeClient:
            def get_option_contracts(self, req):
                calls.append(getattr(req, "page_token", None))
                return FakeResponse(*pages[len(calls) - 1])

        resolver = ChainResolver.__new__(ChainResolver)
        resolver.trading_client = FakeClient()
        got = resolver._fetch_all_pages(SimpleNamespace(page_token=None))

        self.assertEqual(got, ["a", "b", "c", "d", "e"])
        self.assertEqual(len(calls), 3)

    def test_stops_at_the_page_cap_rather_than_looping_forever(self):
        from data_ingestion.chain_resolver import ChainResolver

        class FakeResponse:
            option_contracts = ["x"]
            next_page_token = "always-more"   # a server that never terminates

        class FakeClient:
            def get_option_contracts(self, req):
                return FakeResponse()

        resolver = ChainResolver.__new__(ChainResolver)
        resolver.trading_client = FakeClient()
        got = resolver._fetch_all_pages(SimpleNamespace(page_token=None), max_pages=5)
        self.assertEqual(len(got), 5)


    def test_refuses_to_truncate_silently_when_the_token_cannot_be_set(self):
        """Returning page one quietly is the original bug. Fail loudly instead."""
        from data_ingestion.chain_resolver import ChainResolver

        class FakeResponse:
            option_contracts = ["only-page-one"]
            next_page_token = "more"

        class FakeClient:
            def get_option_contracts(self, req):
                return FakeResponse()

        class Unsettable:
            __slots__ = ()          # cannot accept page_token

        resolver = ChainResolver.__new__(ChainResolver)
        resolver.trading_client = FakeClient()
        with self.assertRaises(RuntimeError):
            resolver._fetch_all_pages(Unsettable())


if __name__ == "__main__":
    unittest.main(verbosity=2)
