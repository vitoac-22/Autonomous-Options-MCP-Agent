"""
The agent loop, wired end to end.

Until now the modules around this file existed but nothing imported them: the
pipeline resolved contracts through a filter that "SPY" defeats, fed a constant
0.1 delta and a flat $2.50 price into the optimiser, and posted an order with no
risk check in between. The scheduled workflow ran exactly that path.

This connects the pieces that were already built and tested:

    config              strategy parameters, expiry band
    chain_resolver      one expiry chosen first, then every strike inside it
    options_market_data real Greeks, IV, quotes, liquidity
    risk_gates          approve / veto / size — the last word before the broker
    mleg_payload        a valid multi-leg order
    decision_journal    every decision recorded, vetoes included

Sizing: leg ratios are fixed by the structure (a condor is 1:1:1:1), and the
gates decide how many of it to execute. See the note at step 7 for why the
convex optimiser is not the right tool for a fixed-shape structure.
"""
import sys
import logging
from dataclasses import replace
from datetime import datetime

import pytz

from config import StrategyConfig
from data_ingestion.alpaca_ingestor import UnderlyingIngestor, OptionsContractResolver
from data_ingestion.chain_resolver import ChainResolver, LegSpec
from data_ingestion.options_market_data import OptionsMarketData
from quant_core.garch_engine import GarchVolatilityEngine
from quant_core.options_pricer import OptionsStrategySelector
from quant_core.strike_mapper import VolatilityCartographer
from quant_core.exit_engine import PortfolioRiskManager
from quant_core.risk_gates import (
    TradeProposal, Position, AccountSnapshot, GateConfig, evaluate,
)
from quant_core.decision_journal import journal_entry, append_entry, DEFAULT_JOURNAL
from ai_agent.options_agent import OptionsExecutionAgent
from ai_agent.mleg_payload import build_mleg_payload, net_limit_price

NY = pytz.timezone("America/New_York")


def market_state(trading_client):
    """Whether the market is open, and how long until the close.

    Read here and handed to the gates, which never touch a clock themselves.
    """
    clock = trading_client.get_clock()
    now_ny = clock.timestamp.astimezone(NY) if clock.timestamp else datetime.now(NY)
    minutes_to_close = 0
    if clock.is_open and clock.next_close:
        minutes_to_close = int(
            (clock.next_close.astimezone(NY) - now_ny).total_seconds() // 60)
    return bool(clock.is_open), max(minutes_to_close, 0), now_ny


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    try:
        cfg = StrategyConfig.from_env()
        logger.info(f"Pipeline start | underlying={cfg.underlying} "
                    f"dte target={cfg.target_dte} band=[{cfg.min_dte},{cfg.max_dte}]")

        # 0. Reconcile existing exposure before deploying any more capital.
        resolver = OptionsContractResolver(underlying_ticker=cfg.underlying)
        open_positions = resolver.trading_client.get_all_positions()
        live_legs = [p for p in open_positions
                     if p.symbol.startswith(cfg.underlying) and len(p.symbol) > 5]

        if live_legs:
            logger.warning(f"Existing exposure: {len(live_legs)} live legs.")
            should_exit, reason = PortfolioRiskManager(live_legs).evaluate_exit_conditions()
            logger.info(f"Exit engine: {reason}")
            if should_exit:
                OptionsExecutionAgent().liquidate_portfolio(live_legs)
            logger.info("=== MONITORING CYCLE COMPLETE ===")
            sys.exit(0)

        logger.info("Portfolio clean. Authorising deployment.")

        # 1. Underlying history.
        today = datetime.now().date()
        ingestor = UnderlyingIngestor(ticker=cfg.underlying, start_date='2016-01-01',
                                      end_date=today.strftime('%Y-%m-%d'))
        log_returns = ingestor.process_memory_data()
        spot = ingestor.get_latest_price()

        # 2. GARCH volatility forecast.
        garch = GarchVolatilityEngine(log_returns)
        _, params, model = garch.fit_model()
        dynamic_var, projected_vol = garch.calculate_tail_risk(model)
        logger.info(f"VaR(99%)={dynamic_var * 100:.2f}% | daily vol={projected_vol * 100:.2f}%"
                    f" | spot=${spot:.2f}")

        # 3+4. Regime -> structure -> theoretical strikes.
        cartographer = VolatilityCartographer(current_price=spot,
                                              garch_vol_daily=projected_vol,
                                              target_dte=cfg.target_dte)
        strategy = OptionsStrategySelector(var_threshold=cfg.var_threshold) \
            .determine_strategy(dynamic_var)
        logger.info(f"Regime {strategy['regime']} -> {strategy['strategy']}")

        strategy['legs'] = (cartographer.map_iron_condor_strikes()
                            if strategy['strategy'] == 'iron_condor'
                            else cartographer.map_straddle_strikes())
        sleeve = "core" if strategy['strategy'] == 'iron_condor' else "convex"

        # 5. One expiry for the whole structure, then every strike inside it.
        specs = [LegSpec(kind=leg['type'], side=leg['side'],
                         target_strike=float(leg['target_strike']))
                 for leg in strategy['legs']]

        final_expiry = (datetime.strptime(cfg.final_expiry, "%Y-%m-%d").date()
                        if cfg.final_expiry else None)
        expiry, contracts = ChainResolver(cfg.underlying).resolve_structure(
            specs, today=today, target_dte=cfg.target_dte,
            min_dte=cfg.min_dte, max_dte=cfg.max_dte,
            not_after=final_expiry,
        )
        logger.info(f"Expiry {expiry} | " +
                    " ".join(f"{c.kind[0].upper()}{c.strike:g}" for c in contracts))

        # 6. Real Greeks, quotes and liquidity — no proxies.
        legs = OptionsMarketData().build_legs(
            [(c.symbol, spec.side, 1) for c, spec in zip(contracts, specs)])
        logger.info("Deltas " + " ".join(f"{leg.delta:+.3f}" for leg in legs) +
                    " | mids " + " ".join(f"{leg.mid:.2f}" for leg in legs))

        # 7. Leg ratios.
        #
        # An iron condor and a straddle are FIXED-ratio structures: 1:1:1:1 and
        # 1:1. Changing the ratios makes it a different strategy with a
        # different risk profile, so delta neutrality has to come from strike
        # SELECTION, not from bending the shape.
        #
        # Running DeltaNeutralOptimizer over free ratios here produced
        # 1:1:1:7 on real deltas — seven long wings against one short call,
        # which is not a condor. It only looked sane before because every leg
        # carried the same fabricated 0.1 delta, so the objective was flat.
        # quant_core/options_allocator.py is left in the repo for variable-ratio
        # structures; it is not the right tool for these two.
        account = resolver.trading_client.get_account()
        equity = float(account.equity)
        ratios = [1] * len(legs)
        legs = [replace(leg, ratio=r) for leg, r in zip(legs, ratios)]
        logger.info(f"Leg ratios {ratios} (fixed — structure shape is not negotiable)")

        # 8. The gates decide whether, and how many.
        proposal = TradeProposal(
            underlying=cfg.underlying, structure=strategy['strategy'],
            sleeve=sleeve, legs=tuple(legs), rationale=strategy['rationale'],
        )
        is_open, minutes_left, now_ny = market_state(resolver.trading_client)
        snapshot = AccountSnapshot(
            equity=equity,
            starting_equity=cfg.contest_start_equity,
            day_pnl=equity - float(account.last_equity or equity),
            now=now_ny.replace(tzinfo=None),
            market_open=is_open,
            minutes_to_close=minutes_left,
            open_positions=tuple(
                Position(underlying=cfg.underlying,
                         risk=abs(float(p.cost_basis)), sleeve=sleeve)
                for p in open_positions
            ),
        )
        verdict = evaluate(proposal, snapshot,
                           GateConfig(min_dte=cfg.min_dte, max_dte=cfg.max_dte,
                                      core_fraction=cfg.core_fraction,
                                      convex_fraction=cfg.convex_fraction))

        append_entry(DEFAULT_JOURNAL, journal_entry(proposal, verdict, snapshot))

        if not verdict.approved:
            logger.warning("VETOED: " + ", ".join(verdict.reasons))
            logger.info("=== CYCLE COMPLETE (no trade) ===")
            sys.exit(0)

        logger.info(f"APPROVED {verdict.contracts}x {strategy['strategy']} | "
                    f"max loss/contract ${verdict.max_loss_per_contract:.2f} | "
                    f"net delta {verdict.net_delta:+.3f}")

        # 9. A valid multi-leg order, priced from real quotes.
        payload = build_mleg_payload(legs, contracts=verdict.contracts,
                                     limit_price=net_limit_price(legs))
        
        # The LLM audits and releases the order through the MCP dispatch tool.
        # submit_mleg_payload raises unless the broker accepted it, so a
        # rejection can no longer be logged as a completed cycle.
        result = OptionsExecutionAgent().submit_mleg_payload(
            payload,
            rationale=strategy.get('rationale', 'Approved by risk gates'),
        )
        logger.info(f"ORDER ACCEPTED by broker | id={result.get('order_id')}")

        logger.info("=== CYCLE COMPLETE ===")

    except Exception:
        # Log the full traceback and exit non-zero so CI reports the failure.
        # Previously this swallowed every error and exited 0, which made GitHub
        # Actions show a green check on runs where no order was ever placed.
        logger.exception("Critical pipeline failure")
        sys.exit(1)
