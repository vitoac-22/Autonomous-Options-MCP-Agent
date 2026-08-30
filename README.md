# Autonomous Options MCP Agent

An options trading agent for the Alpaca AI Trading Agents Hackathon. It models
volatility stochastically rather than reading moving averages, and structures
multi-leg options positions from that forecast.

Everything runs in Alpaca's **paper trading** environment. Paper trading is a
simulation: no real securities are transacted and no real funds move. Nothing
here is investment advice.

---

## How it works

```
GARCH(1,1) volatility forecast          quant_core/garch_engine.py
        │
        ▼
regime selection from dynamic VaR       quant_core/options_pricer.py
        │
        ▼
strike geometry from expected move      quant_core/strike_mapper.py
        │
        ▼
resolve real OCC contracts              data_ingestion/alpaca_ingestor.py
        │
        ▼
real Greeks, IV, quotes, liquidity      data_ingestion/options_market_data.py
        │
        ▼
convex sizing (integer programming)     quant_core/options_allocator.py
        │
        ▼
risk gates — approve / veto / size      quant_core/risk_gates.py
        │
        ▼
atomic multi-leg order                  ai_agent/mleg_payload.py
        │
        ▼
decision journal                        quant_core/decision_journal.py
```

### The volatility engine

Fits a **GARCH(1,1)** model with skewed-t residuals to SPY daily log returns and
forecasts one-step-ahead conditional variance. Dynamic 99% VaR comes from the
empirical quantile of the standardised residuals — filtered historical
simulation, rather than assuming normality.

That VaR selects the regime: compressed volatility structures an **iron condor**,
a VaR breach structures a **long straddle**.

### Strike geometry

Expected move is `σ_daily × √DTE`. Condor wings sit at 1.0 and 1.5 expected
moves either side of spot, so the structure's width derives from the model's own
volatility estimate rather than from fixed offsets.

### Risk gates

`quant_core/risk_gates.py` is a **pure function** — no network, no model, and it
never reads the clock; time and market state arrive on the snapshot. Same inputs
always produce the same verdict, which is what makes the layer exhaustively
testable.

It sits between every decision and the broker:

| Gate | Rule |
|---|---|
| Defined risk | rejects naked short exposure |
| Bid-ask spread | rejects if `(ask−bid)/mid > 10%` on any leg |
| Liquidity | open interest ≥ 500, volume ≥ 100, every leg |
| Expiry window | rejects outside the configured DTE band |
| Delta neutrality | rejects if net delta exceeds the band |
| Daily loss | halts new entries past a daily loss threshold |
| Drawdown floor | disables the convex sleeve, keeps the core running |
| Market hours | no entries outside RTH or near the close |
| Concentration | caps risk per underlying |
| Sleeve budget | enforces the core/convex split |
| Max concurrent | caps open positions |
| Idempotency | a deterministic key prevents double submission |

**Position sizing lives in the gates**, derived from the sleeve budget and the
structure's real maximum loss. A proposal never carries its own contract count,
so no upstream component — a model included — can choose its own size.

Reasons accumulate rather than short-circuiting, so a declined trade reports
every reason it was declined, not only the first.

### Execution

Positions go out as a single **`mleg`** order: atomic by construction, so there
is no partial-fill window and no rollback path to get wrong. Leg ratios are
reduced by their greatest common divisor, which is exact and satisfies Alpaca's
requirement that ratios be in lowest terms. Orders are priced as limits from
real quote mids — a market order across four legs pays the spread four times.

### State and exits

Each run reconciles open positions before deploying capital. `quant_core/exit_engine.py`
evaluates take-profit, stop-loss, and time-decay conditions, liquidating as
expiry approaches — short-dated options move fastest against a position.

Every decision, approved or vetoed and with its reasoning, is appended to
`data/decisions.jsonl` and committed back by the workflow, so the dashboard
reads the same record the agent wrote.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca paper keys
python main_pipeline.py     # one cycle
streamlit run dashboard.py  # the terminal
```

Tests need no credentials and no network:

```bash
python -m unittest discover -p "test_*.py"
```

### Configuration

`config.py` reads these, falling back to defaults:

| Variable | Default | Meaning |
|---|---|---|
| `UNDERLYING` | `SPY` | Traded underlying |
| `TARGET_DTE` | `7` | Target days to expiry |
| `MIN_DTE` / `MAX_DTE` | `1` / `10` | Accepted expiry band |
| `VAR_THRESHOLD` | `-0.035` | VaR level that flips the regime |
| `CORE_FRACTION` / `CONVEX_FRACTION` | `0.60` / `0.40` | Sleeve budgets |

Automation runs via GitHub Actions on a weekday schedule; secrets are
`ALPACA_API_KEY` and `ALPACA_API_SECRET`.

---

## Status

Honest accounting of what is and is not wired up.

**Working:** GARCH engine · regime selection · strike geometry · OCC contract
resolution · real Greeks, IV and quotes · convex sizing · risk gates · mleg
payload construction · exit engine · decision journal · Streamlit dashboard.

**Open:**

- **`mcp_server.py`** exposes the volatility engine over MCP, but nothing
  currently calls it, and Alpaca's own MCP server is not yet in the execution
  path.
- **No LLM is currently connected.** An earlier version routed orders through
  one, but it only reformatted values that were already computed, so it was
  removed. Reconnecting one is worthwhile where a model does something a numeric
  screen cannot — judging whether an implied-versus-realised volatility gap is
  explained by a scheduled catalyst.
- **`BlackScholesPricer`** is implemented but never called. It would be useful
  for sanity-checking market quotes against theoretical value.
- **Sleeve allocation** is defined in config but not yet split across two
  concurrent books.

---

## Stack

Python 3.12 · `alpaca-py` · `arch` (GARCH) · `cvxpy` (integer programming) ·
`streamlit` · `mcp`

## Disclosures

Paper trading is a simulation and does not involve real securities transactions
or real funds. Paper-trading results are hypothetical and do not represent
actual trading or guarantee future results. Options trading carries significant
risk and is not suitable for all investors; see the
[Characteristics and Risks of Standardized Options](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document).
Nothing in this repository is investment advice.

## License

MIT