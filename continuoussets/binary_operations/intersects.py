from __future__ import annotations

from typing import Union, TYPE_CHECKING
import numpy as np
from scipy.optimize import linprog

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry

from continuoussets.unary_operations.convert import Convert

from continuoussets.binary_operations.contains import Contains
from continuoussets.binary_operations.minkowski_sum import MinkowskiSum
from continuoussets.binary_operations.intersection import Intersection

if __name__ == '__main__':
    print('This is the Intersects class.')


@StrategyRegistry
class Intersects(IBinaryOperation):

    # does the order of operands matter?
    ordered = False
    
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 rtol: float, atol: float):
        super().__init__(S1, S2, rtol = rtol, atol = atol)

        # read out concrete implementation
        self = self.select_binary_strategy()

    def __call__(self) -> bool:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@Intersects.register_strategy((SetPair('Interval', 'ndarray'),
                               SetPair('Zonotope', 'ndarray'),
                               SetPair('VPolytope', 'ndarray'),
                               SetPair('HPolyhedron', 'ndarray')))
def _intersects_iconvexset_point(S1, s2, rtol, atol) -> bool:
    return Contains(S1, s2, rtol = rtol, atol = atol)()


@Intersects.register_strategy(SetPair('Interval', 'Interval'))
def _intersects_interval_interval(I1, I2, rtol, atol) -> bool:
    return np.any(np.logical_not(np.any(np.vstack((I2.ub <= I1.lb + atol, I2.lb >= I1.ub - atol)), axis=0)))


@Intersects.register_strategy(SetPair('Zonotope', 'Interval'))
def _intersects_zonotope_interval(Z1, I2, rtol, atol) -> bool:
    Z2 = Convert(I2, 'Zonotope', mode = 'exact')()
    return _intersects_zonotope_zonotope(Z1, Z2, rtol, atol)


@Intersects.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _intersects_zonotope_zonotope(Z1, Z2, rtol, atol) -> bool:
    # use identity: Z1 intersects Z2 iff 0 in Z1 + (-Z2)
    return Contains(MinkowskiSum(Z1, -Z2, mode = 'exact')(), np.zeros(Z1.dimension), rtol = rtol, atol = atol)()


@Intersects.register_strategy(SetPair('VPolytope', 'Interval'))
def _intersects_vpolytope_interval(VP1, I2, rtol, atol) -> bool:
    # read out dimension and number of vertices
    n, m = VP1.dimension, VP1.number_vertices()

    # convert interval to halfspace representation
    HP2 = Convert(I2, 'HPolyhedron', mode = 'exact')()

    # formulate intersection check as linear program: intersection if LP is feasible
    c = np.zeros(m + n)
    A_eq = np.vstack((np.hstack((VP1.V.T, -np.eye(n))),
                      np.hstack((np.ones((1, m)), np.zeros((1, n))))))
    b_eq = np.hstack((np.zeros(n), 1.))
    A_ub = np.vstack((np.hstack((-np.eye(m), np.zeros((m, n)))),
                      np.hstack((np.zeros((2*n, m)), HP2.A))))
    b_ub = np.hstack((np.zeros(m), HP2.b))
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

    return res.success


@Intersects.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _intersects_vpolytope_zonotope(VP1, Z2, rtol, atol) -> bool:
    # read out dimension, number of vertices, number of generators
    m, g = VP1.number_vertices(), Z2.number_generators()

    # formulate intersection check as linear program: intersection if LP is feasible
    c = np.zeros(g + m)
    A_eq = np.vstack((np.hstack((Z2.G.T, -VP1.V.T)),
                      np.hstack((np.zeros((1, g)), np.ones((1, m))))))
    b_eq = np.hstack((-Z2.c, 1.))
    A_ub = np.vstack((np.hstack((np.eye(g), np.zeros((g, m)))),
                      np.hstack((-np.eye(g), np.zeros((g, m)))),
                      np.hstack((np.zeros((m, g)), -np.eye(m)))))
    b_ub = np.hstack((np.ones(2*g), np.zeros(m)))
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

    return res.success


@Intersects.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _intersects_vpolytope_vpolytope(VP1, VP2, rtol, atol) -> bool:
    # special cases can be solved by containment
    m1, m2 = VP1.number_vertices(), VP2.number_vertices()
    if (m1 == 1):
        return Contains(VP2, VP1.V[0], rtol = rtol, atol = atol)()
    elif (m2 == 1):
        return Contains(VP1, VP2.V[0], rtol = rtol, atol = atol)()

    # formulate intersection check as linear program: intersection if LP is feasible
    c = np.zeros(m1 + m2)
    A_eq = np.vstack((np.hstack((VP1.V.T, -VP2.V.T)),
                      np.hstack((np.ones(m1), np.zeros(m2))),
                      np.hstack((np.zeros(m1), np.ones(m2)))))
    b_eq = np.hstack((np.zeros(VP1.dimension), np.array([1., 1.])))
    A_ub = -np.eye(m1 + m2)
    b_ub = np.zeros(m1 + m2)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

    return res.success


@Intersects.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _intersects_hpolyhedron_interval(HP1, I2, rtol, atol) -> bool:
    # linear program: min 0  s.t.  Ax <= b, lb <= x <= ub
    res = linprog(np.zeros(HP1.dimension),
                  A_ub = np.vstack((HP1.A, np.eye(HP1.dimension), -np.eye(HP1.dimension))),
                  b_ub = np.hstack((HP1.b, I2.ub, -I2.lb)),
                  bounds = (None, None))
    return res.success


@Intersects.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _intersects_hpolyhedron_zonotope(HP1, Z2, rtol, atol) -> bool:
    # linear program: min 0  s.t.  Ax <= b, c + Gbeta == x, ||beta||_oo <= 1
    n, m = HP1.dimension, Z2.number_generators()

    c = np.zeros(n + m)
    A_ub = np.vstack((np.hstack((HP1.A, np.zeros((HP1.number_constraints(), m)))),
                      np.hstack((np.zeros((2*m, n)), np.vstack((np.eye(m), -np.eye(m)))))))
    b_ub = np.hstack((HP1.b, np.ones(2*m)))
    A_eq = np.hstack((-np.eye(n), Z2.G.T))
    b_eq = -Z2.c
    res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = (None, None))

    return res.success


@Intersects.register_strategy(SetPair('HPolyhedron', 'VPolytope'))
def _intersects_hpolyhedron_vpolytope(HP1, VP2, rtol, atol) -> bool:
    # linear program: min 0  s.t.  Ax <= b, Vbeta == x, sum beta = 1, beta >= 0
    n, h, m = HP1.dimension, HP1.number_constraints(), VP2.number_vertices()

    c = np.zeros(n + m)
    A_ub = np.vstack((np.hstack((HP1.A, np.zeros((h, m)))),
                      np.hstack((np.zeros((m, n)), -np.eye(m)))))
    b_ub = np.hstack((HP1.b, np.zeros(m)))
    A_eq = np.vstack((np.hstack((-np.eye(n), VP2.V.T)),
                      np.hstack((np.zeros(n), np.ones(m)))))
    b_eq = np.hstack((np.zeros(n), 1.))
    res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = (None, None))

    return res.success


@Intersects.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _intersects_hpolyhedron_hpolyhedron(HP1, HP2, rtol, atol) -> bool:
    return not Intersection(HP1, HP2, mode = 'exact')().empty()
