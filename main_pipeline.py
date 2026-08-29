import sys
import logging
from datetime import datetime
from data_ingestion.alpaca_ingestor import UnderlyingIngestor, OptionsContractResolver
from quant_core.garch_engine import GarchVolatilityEngine
from quant_core.options_pricer import OptionsStrategySelector
from quant_core.options_allocator import DeltaNeutralOptimizer
from quant_core.strike_mapper import VolatilityCartographer
from quant_core.exit_engine import PortfolioRiskManager
from ai_agent.options_agent import OptionsExecutionAgent

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Iniciando Pipeline Institutional Options Alpha Agent")
        
        # 0. Conciliación de Estado (State Management & Exit Engine)
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        posiciones_abiertas = resolver.trading_client.get_all_positions()
        patas_activas = [p for p in posiciones_abiertas if p.symbol.startswith('SPY') and len(p.symbol) > 5]
        
        if patas_activas:
            logger.warning(f"Exposición detectada: {len(patas_activas)} patas vivas.")
            
            risk_manager = PortfolioRiskManager(patas_activas)
            debe_liquidar, razon = risk_manager.evaluate_exit_conditions()
            
            if debe_liquidar:
                logger.warning(f"Veredicto del Motor de Riesgo: {razon}")
                agente = OptionsExecutionAgent()
                agente.liquidate_portfolio(patas_activas)
            else:
                logger.info(f"Veredicto del Motor de Riesgo: {razon}")
                
            logger.info("=== CICLO DE MONITOREO COMPLETADO ===")
            sys.exit(0)
            
        logger.info("Portafolio limpio. Autorizando despliegue de capital.")
        
        # 1. Ingestión del Subyacente
        hoy_str = datetime.now().strftime('%Y-%m-%d')
        ingestor = UnderlyingIngestor(ticker='SPY', start_date='2016-01-01', end_date=hoy_str)
        retornos_log = ingestor.process_memory_data()
        precio_actual_spy = ingestor.get_latest_price()
        
        # 2. Motor GARCH
        motor_garch = GarchVolatilityEngine(retornos_log)
        df_res, parametros, modelo = motor_garch.fit_model()
        var_dinamico, vol_proyectada = motor_garch.calculate_tail_risk(modelo)
        logger.info(f"VaR(99%) proyectado: {var_dinamico * 100:.2f}% | Volatilidad Diaria: {vol_proyectada * 100:.2f}% | Precio Spot: ${precio_actual_spy}")
        
        # 3. Cartografía de Volatilidad
        cartografo = VolatilityCartographer(current_price=precio_actual_spy, garch_vol_daily=vol_proyectada, target_dte=30)
        
        # 4. Selección Dinámica de Estrategia
        selector = OptionsStrategySelector()
        estrategia = selector.determine_strategy(var_dinamico)
        logger.info(f"Régimen detectado: {estrategia['regime']} -> Estructurando: {estrategia['strategy']}")
        
        if estrategia['strategy'] == 'iron_condor':
            estrategia['legs'] = cartografo.map_iron_condor_strikes()
        else:
            estrategia['legs'] = cartografo.map_straddle_strikes()
            
        # 5. Resolución OCC
        logger.info("Escaneando cadena de opciones y resolviendo contratos OCC reales...")
        deltas_estimados = []
        precios_estimados = []
        signs_list = []
        used_symbols = set()
        
        for leg in estrategia['legs']:
            occ_symbol = resolver.get_valid_occ_symbol(leg['target_strike'], leg['type'], exclude_symbols=used_symbols)
            used_symbols.add(occ_symbol)
            leg['symbol'] = occ_symbol
            d_proxy = 0.1 if abs(leg['sign']) == 1 else 0.5
            deltas_estimados.append(d_proxy)
            precios_estimados.append(2.5)
            signs_list.append(leg['sign'])
        
        # 6. Optimización Convexa de Capital (Basado en Equity Real)
        account_info = resolver.trading_client.get_account()
        real_equity = float(account_info.equity)
        # Restricción institucional: Arriesgar máximo el 10% del portafolio, con un tope nominal de $10,000.
        margen_asignado = min(real_equity * 0.10, 10000.0) 
        
        logger.info(f"Capital auditado (Equity): ${real_equity:.2f} -> Asignando: ${margen_asignado:.2f}")

        optimizador = DeltaNeutralOptimizer(leg_deltas=deltas_estimados, leg_prices=precios_estimados, leg_signs=signs_list, total_capital=margen_asignado)
        tamanos_contratos = optimizador.optimize_sizing()
        
        for i, leg in enumerate(estrategia['legs']):
            leg['qty'] = tamanos_contratos[i]
            logger.info(f"Pata Estructural [{i}]: {leg['side'].upper()} {leg['qty']}x {leg['symbol']} | Strike Teórico: ${leg['target_strike']} | Signo: {leg['sign']}")
            
        # 7. Ejecución Autónoma MLEG
        agente = OptionsExecutionAgent()
        agente.execute_atomic_transaction(estrategia)
        
        logger.info("=== CICLO DIARIO COMPLETADO CON ÉXITO ===")

    except Exception:
        # Log the full traceback and exit non-zero so CI reports the failure.
        # Previously this swallowed every error and exited 0, which made GitHub
        # Actions show a green check on runs where no order was ever placed.
        logger.exception("Fallo crítico en el pipeline institucional")
        sys.exit(1)
