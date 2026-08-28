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
        logger.info("Iniciando Pipeline Options Alpha Agent")
        
        # 1. Ingesta SRP
        ingestor = UnderlyingIngestor(ticker='SPY', start_date='2016-01-01', end_date='2026-08-28')
        retornos_log = ingestor.process_memory_data()
        precio_actual_spy = ingestor.get_latest_price()
        
        # 2. Motor GARCH SRP
        motor_garch = GarchVolatilityEngine(retornos_log)
        df_res, parametros, modelo = motor_garch.fit_model()
        var_dinamico, vol_proyectada = motor_garch.calculate_tail_risk(modelo)
        logger.info(f"VaR(99%) proyectado: {var_dinamico * 100:.2f}% | Volatilidad Diaria: {vol_proyectada * 100:.2f}% | Precio Spot: ${precio_actual_spy}")
        
        # 3. Cartografía de Volatilidad
        cartografo = VolatilityCartographer(current_price=precio_actual_spy, garch_vol_daily=vol_proyectada, target_dte=30)
        
        # 4. Selección de Estrategia OCP
        selector = OptionsStrategySelector()
        estrategia = selector.determine_strategy(var_dinamico)
        logger.info(f"Régimen: {estrategia['regime']} -> Estructurando: {estrategia['strategy']}")
        
        # 5. Resolución de Contratos Vivos en Alpaca (Consistencia ACID)
        resolver = OptionsContractResolver(underlying_ticker='SPY')
        
        if estrategia['strategy'] == 'iron_condor':
            strikes = cartografo.map_iron_condor_strikes()
            estrategia['legs'][0]['symbol'] = resolver.get_valid_occ_symbol(strikes['long_put'], 'put')
            estrategia['legs'][1]['symbol'] = resolver.get_valid_occ_symbol(strikes['short_put'], 'put')
            estrategia['legs'][2]['symbol'] = resolver.get_valid_occ_symbol(strikes['short_call'], 'call')
            estrategia['legs'][3]['symbol'] = resolver.get_valid_occ_symbol(strikes['long_call'], 'call')
        elif estrategia['strategy'] == 'long_straddle':
            strikes = cartografo.map_straddle_strikes()
            estrategia['legs'][0]['symbol'] = resolver.get_valid_occ_symbol(strikes['long_call'], 'call')
            estrategia['legs'][1]['symbol'] = resolver.get_valid_occ_symbol(strikes['long_put'], 'put')

        # 6. Asignación Convexa ACID
        deltas_simulados = [0.5, -0.5] if estrategia['strategy'] == 'long_straddle' else [0.1, -0.1, -0.1, 0.1]
        precios_simulados = [5.0, 5.0] if estrategia['strategy'] == 'long_straddle' else [1.0, 2.0, 2.0, 1.0]
        
        optimizador = DeltaNeutralOptimizer(deltas_simulados, precios_simulados, total_capital=10000.0)
        tamanos_contratos = optimizador.optimize_sizing()
        
        for i, leg in enumerate(estrategia['legs']):
            leg['qty'] = tamanos_contratos[i]
            
        # 7. Ejecución Autónoma
        agente = OptionsExecutionAgent()
        agente.execute_atomic_transaction(estrategia)
        
        logger.info("=== CICLO DIARIO COMPLETADO ===")

    except Exception as e:
        logger.error(f"Fallo de infraestructura: {str(e)}")