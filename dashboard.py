import streamlit as st
import pandas as pd
from datetime import datetime, time
import pytz
import os
from alpaca.trading.requests import GetPortfolioHistoryRequest
from data_ingestion.alpaca_ingestor import OptionsContractResolver
from quant_core.decision_journal import (
    read_entries, summarise, DEFAULT_JOURNAL,
)

# Institutional Page Configuration
st.set_page_config(page_title="AlphaOptions Terminal", layout="wide", initial_sidebar_state="collapsed")

# Injected CSS for professional terminal aesthetics
st.markdown("""
    <style>
    .big-font { font-size:22px !important; font-weight: 600; color: #4CAF50; }
    .engine-font { font-size:22px !important; font-weight: 600; color: #29B6F6; }
    .status-font { font-size:20px !important; font-weight: 600; }
    .stDataFrame { border: 1px solid #333333; }
    </style>
    """, unsafe_allow_html=True)

st.title("Institutional Options Alpha Agent")
st.markdown("---")

# 1. ORCHESTRATION & STATE CLOCK
col1, col2, col3 = st.columns(3)

# Next execution window calculation (15:45 EDT)
ny_tz = pytz.timezone('America/New_York')
now_ny = datetime.now(ny_tz)
target_time = time(15, 45, 0)
execution_today = ny_tz.localize(datetime.combine(now_ny.date(), target_time))

if now_ny > execution_today:
    status = "STANDBY (CYCLE EXECUTED)"
    color = "white"
else:
    status = "ARMED (AWAITING WINDOW)"
    color = "#00E676"

with col1:
    st.subheader("Orchestrator Status")
    st.markdown(f"<span class='status-font' style='color:{color};'>{status}</span>", unsafe_allow_html=True)
    st.caption("Target Execution: 15:45 EDT (19:45 UTC)")

with col2:
    st.subheader("Underlying Asset")
    st.markdown("<span class='big-font'>SPDR S&P 500 ETF (SPY)</span>", unsafe_allow_html=True)

with col3:
    st.subheader("Stochastic Engine")
    st.markdown("<span class='engine-font'>GARCH(1,1) Skewed-t</span>", unsafe_allow_html=True)

st.markdown("---")

# 2. LIVE PORTFOLIO MATRIX & EQUITY CURVE
st.header("Live Exposure & State Management")

@st.cache_data(ttl=60)
def fetch_portfolio_and_history():
    try:
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        client = resolver.trading_client
        
        # Extracción de Equity Real
        account = client.get_account()
        real_equity = float(account.equity)
        
        # Corrección estricta de tipo string para el timeframe del portafolio
        req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        history = client.get_portfolio_history(req)
        
        df_history = pd.DataFrame({
            "Date": [datetime.fromtimestamp(ts).date() for ts in history.timestamp],
            "Equity": [float(e) if e is not None else real_equity for e in history.equity]
        }).set_index("Date")

        # Extracción de Patas Vivas
        positions = client.get_all_positions()
        legs = [p for p in positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if not legs:
            return pd.DataFrame(), real_equity, df_history
            
        data = []
        for p in legs:
            try:
                date_str = p.symbol[3:9]
                exp_date = datetime.strptime(date_str, '%y%m%d').date()
                dte = (exp_date - datetime.now().date()).days
            except:
                dte = "N/A"
                
            data.append({
                "OCC Symbol": p.symbol,
                "Side": p.side.upper(),
                "Qty": int(p.qty),
                "Market Value ($)": float(p.market_value),
                "Unrealized PnL ($)": float(p.unrealized_pl),
                "DTE": dte
            })
        return pd.DataFrame(data), real_equity, df_history
    except Exception as e:
        st.error(f"Clearing house connection failure (Alpaca): {e}")
        return pd.DataFrame(), 0.0, pd.DataFrame()

df_port, equity_val, df_history = fetch_portfolio_and_history()

col_bp, col_risk = st.columns([1, 2])
with col_bp:
    st.metric(label="Net Portfolio Value (Equity)", value=f"${equity_val:,.2f}")
    if not df_history.empty:
        st.line_chart(df_history, y="Equity", color="#00E676")

with col_risk:
    if df_port.empty:
        st.info("Clean portfolio. Gamma risk neutralized. Awaiting next volatility injection signal.")
    else:
        formatted_df = df_port.style.format({
            "Market Value ($)": "{:.2f}",
            "Unrealized PnL ($)": "{:.2f}"
        }).map(
            lambda x: 'color: #00E676' if x > 0 else ('color: #FF5252' if x < 0 else ''),
            subset=['Unrealized PnL ($)']
        )
        st.dataframe(formatted_df, width='stretch', hide_index=True)

st.markdown("---")

# 3. DECISION JOURNAL
#
# This previously tailed "pipeline.log". That file is written by the GitHub
# Actions runner, which is destroyed when the job ends, while this dashboard
# reads its own filesystem — so the panel showed "pending generation" forever in
# production. The decision journal is committed back by the workflow, so both
# sides read the same file.
st.header("Decision Journal")
st.caption("Every decision the agent made, approved or refused, with its reasoning.")

decisions = read_entries(DEFAULT_JOURNAL)

if not decisions:
    st.info("No decisions recorded yet. The orchestrator writes one per cycle.")
else:
    stats = summarise(decisions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions", stats["total"])
    c2.metric("Executed", stats["approved"])
    c3.metric("Refused", stats["vetoed"])
    c4.metric("Contracts", stats["contracts_traded"])

    if stats["top_veto_reasons"]:
        st.caption("Why the agent declined to trade")
        st.dataframe(
            pd.DataFrame(stats["top_veto_reasons"], columns=["Reason", "Times"]),
            width='stretch', hide_index=True,
        )

    st.markdown("---")

    for entry in reversed(decisions[-15:]):
        stamp = str(entry.get("timestamp", ""))[:19].replace("T", " ")
        executed = entry.get("approved")
        badge = "🟢 EXECUTED" if executed else "🔴 REFUSED"
        headline = (f"{badge} — {stamp} — {entry.get('structure', '?')} "
                    f"({entry.get('sleeve', '?')} sleeve)")

        with st.expander(headline, expanded=not executed and entry is decisions[-1]):
            if executed:
                st.success(f"Executed {entry.get('contracts')} contracts")
            else:
                for reason in entry.get("reasons", []):
                    st.error(f"Gate refused: {reason.replace('_', ' ')}")

            st.caption(f"Model rationale: {entry.get('rationale', '—')}")

            m1, m2, m3 = st.columns(3)
            net_delta = entry.get("net_delta")
            max_loss = entry.get("max_loss_per_contract")
            m1.metric("Net delta", f"{net_delta:+.4f}" if net_delta is not None else "—")
            m2.metric("Max loss / contract",
                      f"${max_loss:,.0f}" if max_loss is not None else "—")
            m3.metric("Equity", f"${entry.get('equity', 0):,.0f}")

            legs = entry.get("legs", [])
            if legs:
                st.dataframe(pd.DataFrame([{
                    "Side": leg.get("side", "").upper(),
                    "×": leg.get("ratio"),
                    "Type": leg.get("kind", "").upper(),
                    "Strike": leg.get("strike"),
                    "Expiry": leg.get("expiry"),
                    "Delta": round(leg.get("delta", 0), 4),
                    "Mid": leg.get("mid"),
                    "OI": leg.get("open_interest"),
                    "Contract": leg.get("symbol"),
                } for leg in legs]), width='stretch', hide_index=True)
