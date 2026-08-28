import cvxpy as cp
import numpy as np

class DeltaNeutralOptimizer:
    def __init__(self, leg_deltas: list, leg_prices: list, total_capital: float):
        self.deltas = np.array(leg_deltas)
        self.prices = np.array(leg_prices)
        self.capital = total_capital
        self.n_legs = len(leg_deltas)

    def optimize_sizing(self):
        # Atomicidad (ACID): Los contratos de opciones no son divisibles. Se restringe el solver a enteros.
        w = cp.Variable(self.n_legs, integer=True)
        
        # Objetivo Convexo: Minimizar el Delta neto absoluto del portafolio
        net_delta = cp.sum(cp.multiply(w, self.deltas))
        objective = cp.Minimize(cp.abs(net_delta))
        
        # Restricciones estructurales y de capital (Multiplicador estándar de opciones = 100)
        cost = cp.sum(cp.multiply(w, self.prices)) * 100 
        constraints = [
            cost <= self.capital,
            w >= 1 
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB) 
        
        return [int(val) for val in w.value]