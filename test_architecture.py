import unittest
from quant_core.options_allocator import DeltaNeutralOptimizer
from quant_core.strike_mapper import VolatilityCartographer

class TestInstitutionalArchitecture(unittest.TestCase):

    def test_convex_allocation_iron_condor(self):
        """
        Prueba que el optimizador respete los signos (-1 para ventas, 1 para compras)
        y no quiebre el límite de capital.
        """
        # deltas típicos OTM: Alas (0.1), Centro (0.4)
        deltas = [0.1, 0.4, 0.4, 0.1]
        precios = [1.5, 3.0, 3.0, 1.5]
        signos = [1, -1, -1, 1]  # Long Put, Short Put, Short Call, Long Call
        capital = 5000.0
        
        optimizador = DeltaNeutralOptimizer(deltas, precios, signos, capital)
        tamanos = optimizador.optimize_sizing()
        
        # Validación de reglas institucionales
        self.assertEqual(len(tamanos), 4)
        self.assertTrue(all(qty >= 1 for qty in tamanos))
        
        # Validar consumo de capital nocional
        # W es qty * signo. Abs(W) = qty.
        costo_total = sum(abs(tamanos[i]) * precios[i] * 100 for i in range(4))
        self.assertLessEqual(costo_total, capital)

    def test_strike_mapping_geometry(self):
        """
        Verifica que la cartografía respete el esqueleto matemático del Condor.
        """
        precio_spot = 100.0
        vol_diaria = 0.02 # 2%
        cartografo = VolatilityCartographer(precio_spot, vol_diaria, target_dte=30)
        patas = cartografo.map_iron_condor_strikes()
        
        # El ala protectora inferior (Long Put) debe tener el strike más bajo
        self.assertTrue(patas[0]['target_strike'] < patas[1]['target_strike'])
        # Las ventas de riesgo (Shorts) deben rodear el precio spot
        self.assertTrue(patas[1]['target_strike'] < precio_spot < patas[2]['target_strike'])
        # El ala protectora superior (Long Call) debe ser el strike más alto
        self.assertTrue(patas[2]['target_strike'] < patas[3]['target_strike'])
        # Los signos deben ser simétricos
        self.assertEqual([p['sign'] for p in patas], [1, -1, -1, 1])

if __name__ == '__main__':
    unittest.main()

class TestWingTightening(unittest.TestCase):
    """The cartographer can tighten wings for feeds without far-OTM Greeks."""

    def test_wing_multiplier_moves_only_the_wings(self):
        from quant_core.strike_mapper import VolatilityCartographer
        wide = VolatilityCartographer(100.0, 0.02, target_dte=7).map_iron_condor_strikes()
        tight = VolatilityCartographer(100.0, 0.02, target_dte=7) \
            .map_iron_condor_strikes(wing_mult=1.15)
        # shorts unchanged
        self.assertEqual(wide[1]['target_strike'], tight[1]['target_strike'])
        self.assertEqual(wide[2]['target_strike'], tight[2]['target_strike'])
        # wings strictly closer to spot
        self.assertGreater(tight[0]['target_strike'], wide[0]['target_strike'])
        self.assertLess(tight[3]['target_strike'], wide[3]['target_strike'])
