import logging
from data_ingestion.alpaca_ingestor import UnderlyingIngestor, OptionsContractResolver
from quant_core.garch_engine import GarchVolatilityEngine
from quant_core.options_pricer import OptionsStrategySelector
from quant_core.options_allocator import DeltaNeutralOptimizer
from quant_core.strike_mapper import VolatilityCartographer
from ai_agent.options_agent import OptionsExecutionAgent

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Iniciando Pipeline Institutional Options Alpha Agent")
        
        # 1. Ingestión del Subyacente (Datos históricos de SPY)
        ingestor = UnderlyingIngestor(ticker='SPY', start_date='2016-01-01', end_date='2026-08-28')
        retornos_log = ingestor.process_memory_data()
        precio_actual_spy = ingestor.get_latest_price()
        
        # 2. Motor GARCH (Cálculo de Varianza Condicional y VaR al 99%)
        motor_garch = GarchVolatilityEngine(retornos_log)
        df_res, parametros, modelo = motor_garch.fit_model()
        var_dinamico, vol_proyectada = motor_garch.calculate_tail_risk(modelo)
        logger.info(f"VaR(99%) proyectado: {var_dinamico * 100:.2f}% | Volatilidad Diaria: {vol_proyectada * 100:.2f}% | Precio Spot: ${precio_actual_spy}")
        
        # 3. Cartografía de Volatilidad (Geometría de Strikes y Signos)
        cartografo = VolatilityCartographer(current_price=precio_actual_spy, garch_vol_daily=vol_proyectada, target_dte=30)
        
        # 4. Selección Dinámica de Estrategia basada en el Régimen de Riesgo
        selector = OptionsStrategySelector()
        estrategia = selector.determine_strategy(var_dinamico)
        logger.info(f"Régimen detectado: {estrategia['regime']} -> Estructurando: {estrategia['strategy']}")
        
        # Obtener las patas estructurales con sus direcciones (Long/Short) y strikes teóricos
        if estrategia['strategy'] == 'iron_condor':
            estrategia['legs'] = cartografo.map_iron_condor_strikes()
        else:
            estrategia['legs'] = cartografo.map_straddle_strikes()
            
        # 5. Resolución de Símbolos OCC Reales sin Colisiones
        resolver = OptionsContractResolver(underlying_ticker='SPY')
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
        
        # 6. Optimización Convexa de Capital y Delta Neutral (Respetando Signos de Posición)
        optimizador = DeltaNeutralOptimizer(
            leg_deltas=deltas_estimados, 
            leg_prices=precios_estimados, 
            leg_signs=signs_list, 
            total_capital=10000.0
        )
        tamanos_contratos = optimizador.optimize_sizing()
        
        for i, leg in enumerate(estrategia['legs']):
            leg['qty'] = tamanos_contratos[i]
            logger.info(f"Pata Estructural [{i}]: {leg['side'].upper()} {leg['qty']}x {leg['symbol']} | Strike Teórico: ${leg['target_strike']} | Signo: {leg['sign']}")
            
        # 7. Ejecución Autónoma con Protocolo ACID y SDK Nativo de Alpaca
        agente = OptionsExecutionAgent()
        agente.execute_atomic_transaction(estrategia)
        
        logger.info("=== CICLO DIARIO COMPLETADO CON ÉXITO ===")

    except Exception as e:
        logger.error(f"Fallo crítico en el pipeline institucional: {str(e)}")