import cvxpy as cp
import numpy as np

class DeltaNeutralOptimizer:
    def __init__(self, leg_deltas: list, leg_prices: list, leg_signs: list, total_capital: float):
        """
        leg_signs: Lista de enteros con +1 para posiciones largas (compra) y -1 para cortas (venta).
        """
        self.deltas = np.array(leg_deltas)
        self.prices = np.array(leg_prices)
        self.signs = np.array(leg_signs)
        self.capital = total_capital
        self.n_legs = len(leg_deltas)

    def optimize_sizing(self):
        # Magnitud entera de contratos para cada pata (siempre >= 1 en valor absoluto)
        m = cp.Variable(self.n_legs, integer=True)
        
        # Vector de posición real considerando la dirección de la pata (Long/Short)
        w = cp.multiply(m, self.signs)
        
        # Objetivo: Minimizar el Delta neto absoluto del portafolio multi-pata
        net_delta = cp.sum(cp.multiply(w, self.deltas))
        objective = cp.Minimize(cp.abs(net_delta))
        
        # Costo total del portafolio (las ventas restan capital requerido o generan crédito, 
        # pero para dimensionamiento conservador evaluamos el valor nocional absoluto o neto)
        cost = cp.sum(cp.multiply(cp.abs(w), self.prices)) * 100 
        
        constraints = [
            cost <= self.capital,
            m >= 1,  # Cada pata debe tener al menos 1 contrato operativo
            m <= 10  # Lote máximo por pata para control de riesgo institucional
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Fallo del solver convexo: Estado {problem.status}")
            
        return [int(val) for val in m.value]