import os
import sys
import json
import logging
from datetime import datetime
from mcp.server.mcpserver import MCPServer

# Secuestro de Logs: Silenciamos cualquier salida a consola (STDIO) para no corromper el protocolo MCP
logging.basicConfig(level=logging.WARNING, filename='mcp_silent.log', filemode='w')

from data_ingestion.alpaca_ingestor import UnderlyingIngestor, OptionsContractResolver
from quant_core.garch_engine import GarchVolatilityEngine
from quant_core.exit_engine import PortfolioRiskManager

# Inicialización del Servidor MCP (Estándar v2.x)
mcp = MCPServer("AlphaOptionsAgent")

@mcp.tool()
def get_portfolio_state() -> str:
    """Obtiene el estado actual del portafolio en Alpaca, filtrando opciones de SPY vivas."""
    try:
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        posiciones = resolver.trading_client.get_all_positions()
        patas_activas = [p for p in posiciones if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if not patas_activas:
            return json.dumps({"status": "clean", "positions": []})
            
        data = []
        for p in patas_activas:
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
    """Evalúa las condiciones estructurales de salida (Stop Loss, Take Profit, Riesgo Gamma DTE <= 5)."""
    try:
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        posiciones = resolver.trading_client.get_all_positions()
        patas_activas = [p for p in posiciones if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if not patas_activas:
            return "Portafolio limpio. No hay riesgo Gamma estructural presente."
            
        manager = PortfolioRiskManager(patas_activas)
        debe_liquidar, razon = manager.evaluate_exit_conditions()
        
        return json.dumps({
            "liquidation_required": debe_liquidar,
            "reason": razon
        })
    except Exception as e:
        return f"Error en la evaluación de riesgo: {str(e)}"

@mcp.tool()
def get_volatility_regime() -> str:
    """Ejecuta el motor estocástico GARCH(1,1) para SPY y devuelve el VaR(99%) dinámico."""
    try:
        hoy_str = datetime.now().strftime('%Y-%m-%d')
        ingestor = UnderlyingIngestor(ticker='SPY', start_date='2016-01-01', end_date=hoy_str)
        retornos = ingestor.process_memory_data()
        spot = ingestor.get_latest_price()
        
        motor = GarchVolatilityEngine(retornos)
        df_res, params, modelo = motor.fit_model()
        var, vol = motor.calculate_tail_risk(modelo)
        
        regime = "high_volatility" if var < -0.035 else "volatility_compression"
        
        return json.dumps({
            "spot_price": float(spot),
            "projected_daily_volatility": round(float(vol) * 100, 2),
            "var_99": round(float(var) * 100, 2),
            "regime": regime
        })
    except Exception as e:
        return f"Error crítico en GARCH engine: {str(e)}"

if __name__ == "__main__":
    mcp.run()