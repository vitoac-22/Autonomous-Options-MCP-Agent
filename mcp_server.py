import os
import sys
import json
import logging
import requests
from datetime import datetime
from mcp.server.mcpserver import MCPServer

# Log hijacking: Silence console output to prevent MCP protocol corruption
logging.basicConfig(level=logging.WARNING, filename='mcp_silent.log', filemode='w')

from data_ingestion.alpaca_ingestor import UnderlyingIngestor, OptionsContractResolver
from quant_core.garch_engine import GarchVolatilityEngine
from quant_core.exit_engine import PortfolioRiskManager
from ai_agent.execution_bridge import verify_approval, ApprovalMismatch

# Initialize standard v2.x MCP Server
mcp = MCPServer("AlphaOptionsAgent")

@mcp.tool()
def get_portfolio_state() -> str:
    """Fetches the current portfolio state from Alpaca, filtering live SPY options."""
    try:
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        positions = resolver.trading_client.get_all_positions()
        active_legs = [p for p in positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if not active_legs:
            return json.dumps({"status": "clean", "positions": []})
            
        data = []
        for p in active_legs:
            data.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "unrealized_pl": float(p.unrealized_pl),
                "market_value": float(p.market_value)
            })
        return json.dumps({"status": "exposed", "positions": data})
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def evaluate_gamma_risk() -> str:
    """Evaluates structural exit conditions (Stop Loss, Take Profit, Gamma Risk DTE <= 5)."""
    try:
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        positions = resolver.trading_client.get_all_positions()
        active_legs = [p for p in positions if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if not active_legs:
            return "Clean portfolio. No structural Gamma risk present."
            
        manager = PortfolioRiskManager(active_legs)
        must_liquidate, reason = manager.evaluate_exit_conditions()
        
        return json.dumps({
            "liquidation_required": must_liquidate,
            "reason": reason
        })
    except Exception as e:
        return f"Risk evaluation error: {str(e)}"

@mcp.tool()
def get_volatility_regime() -> str:
    """Runs the GARCH(1,1) stochastic engine for SPY and returns dynamic VaR(99%)."""
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        ingestor = UnderlyingIngestor(ticker='SPY', start_date='2016-01-01', end_date=today_str)
        log_returns = ingestor.process_memory_data()
        spot = ingestor.get_latest_price()
        
        engine = GarchVolatilityEngine(log_returns)
        df_res, params, model = engine.fit_model()
        var, vol = engine.calculate_tail_risk(model)
        
        regime = "high_volatility" if var < -0.035 else "volatility_compression"
        
        return json.dumps({
            "spot_price": float(spot),
            "projected_daily_volatility": round(float(vol) * 100, 2),
            "var_99": round(float(var) * 100, 2),
            "regime": regime
        })
    except Exception as e:
        return f"Critical GARCH engine error: {str(e)}"

@mcp.tool()
def dispatch_mleg_order(payload_json: str, approval_hash: str = "") -> str:
    """
    Dispatch a risk-gate-approved multi-leg order to Alpaca.

    The payload is verified against the approval hash issued by the risk gates
    before anything is sent. The calling LLM authors these arguments, so without
    that check it could alter strikes, sides, ratios or quantities after the
    gates had approved a different structure. The system prompt asks it not to;
    this makes it impossible.
    """
    try:
        payload = verify_approval(payload_json, approval_hash)
    except ApprovalMismatch as exc:
        return json.dumps({"status": "refused", "error": str(exc)})

    try:
        headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY"),
            "Content-Type": "application/json",
        }
        response = requests.post("https://paper-api.alpaca.markets/v2/orders",
                                 json=payload, headers=headers, timeout=30)

        if response.status_code in (200, 201):
            return json.dumps({"status": "success", "order_id": response.json().get("id")})

        return json.dumps({"status": "rejected", "error": response.text})
    except Exception as exc:
        return json.dumps({"status": "critical_failure", "error": str(exc)})

if __name__ == "__main__":
    mcp.run()
