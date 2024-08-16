from __future__ import annotations

from typing import TYPE_CHECKING, Union, Dict, Tuple, Callable
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

import continuoussets.convexsets.interval as interval
import continuoussets.convexsets.zonotope as zonotope
import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.unary_operations.convert import Convert

from continuoussets.utils.auxiliary import SetPair

if __name__ == '__main__':
    print('This is the MinkowskiSum class.')


class MinkowskiSum(IBinaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is relevant
    ordered_operation = False
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):

        # call superclass constructor
        super().__init__(S1, S2, mode)

        # get concrete implementation function
        strategy_key = self.get_strategy_key()
        self.func = MinkowskiSum.strategies.get(strategy_key, None)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError
        
    def __call__(self) -> Union[np.ndarray, 'IConvexSet']:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@MinkowskiSum.register_strategy((SetPair('ndarray', 'ndarray'),
                                 SetPair('Interval', 'ndarray'),
                                 SetPair('Zonotope', 'ndarray'),
                                 SetPair('VPolytope', 'ndarray'),
                                 SetPair('HPolyhedron', 'ndarray')))
def _minkowski_sum_any_point(S1, s2, mode) -> np.ndarray:
    return S1 + s2


@MinkowskiSum.register_strategy(SetPair('Interval', 'Interval'))
def _minkowski_sum_interval_interval(I1, I2, mode) -> interval.Interval:
    return interval.Interval(lb = I1.lb + I2.lb,
                             ub = I1.ub + I2.ub,
                             validate = False)


@MinkowskiSum.register_strategy(SetPair('Zonotope', 'Interval'))
def _minkowski_sum_zonotope_interval(Z1, I2, mode) -> zonotope.Zonotope:
    Z2 = Convert(I2, 'Zonotope', mode = 'exact')
    return _minkowski_sum_zonotope_zonotope(Z1, Z2, mode = mode)


@MinkowskiSum.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _minkowski_sum_zonotope_zonotope(Z1, Z2, mode) -> zonotope.Zonotope:
    return zonotope.Zonotope(c = Z1.c + Z2.c,
                             G = np.vstack((Z1.G, Z2.G)),
                             validate = False)


@MinkowskiSum.register_strategy((SetPair('VPolytope', 'Interval'),
                                 SetPair('VPolytope', 'Zonotope')))
def _minkowski_sum_vpolytope_other(VP1, S2, mode) -> vpolytope.VPolytope:
    VP2 = Convert(S2, 'VPolytope', mode = mode)
    return _minkowski_sum_vpolytope_vpolytope(VP1, VP2, mode = mode)


@MinkowskiSum.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _minkowski_sum_vpolytope_vpolytope(VP1, VP2, mode) -> vpolytope.VPolytope:
    # add each combination
    V_sum = np.zeros((VP1.number_vertices()*VP2.number_vertices(), VP1.dimension))
    # todo: replace this by a faster method
    for i in range(VP1.number_vertices()):
        for j in range(VP2.number_vertices()):
            V_sum[i * VP2.number_vertices() + j] = VP1.V[i] + VP2.V[j]

    return vpolytope.VPolytope(V = V_sum, validate = False)


@MinkowskiSum.register_strategy((SetPair('HPolyhedron', 'Interval'),
                                 SetPair('HPolyhedron', 'Zonotope'),
                                 SetPair('HPolyhedron', 'VPolytope')))
def _minkowski_sum_hpolyhedron_other(HP1, I2, mode) -> hpolyhedron.HPolyhedron:
    HP2 = Convert(I2, 'HPolyhedron', mode = mode)
    return _minkowski_sum_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)


@MinkowskiSum.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _minkowski_sum_hpolyhedron_hpolyhedron(HP1, HP2, mode) -> hpolyhedron.HPolyhedron:
    if mode == 'inner':
        return _minkowski_sum_hpolyhedron_hpolyhedron(HP1, HP2, mode = 'exact')
    
    if mode == 'exact':
        n1, h1 = HP1.dimension, HP1.number_constraints()
        n2, h2 = HP2.dimension, HP2.number_constraints()

        # lift and project onto first n dimensions (rewriting of Cartesian product...)
        HP_lifted = hpolyhedron.HPolyhedron(A = np.block([[HP1.A, np.zeros((n2, h1))], [np.zeros((n1, h2)), HP2.A]]),
                                            b = np.hstack(HP1.b, HP2.b),
                                            validate = False)
        M = np.hstack((np.eye(HP1.dimension), np.eye(HP1.dimension)))
        return HP_lifted.matmul(M)
    
    if mode == 'outer':
        n, h = HP1.dimension, HP1.number_constraints()

        # addition of support function evaluation
        A_new = np.vstack((HP1.A, np.eye(n), -np.eye(n)))
        b_new = np.hstack((HP1.b, np.zeros(2*n)))
    
        # first h constraints: only compute support function of HP2
        for i in range(h):
            b_new[i] += HP2.support_function(A_new[i])[0]

        # remaining 2n constraints: also compute support function of HP1
        for i in range(2*n):
            b_new[h+i] = HP1.support_function(A_new[h+i])[0] + HP2.support_function(A_new[h+i])[0]

        return hpolyhedron.HPolyhedron(A = A_new, b = b_new, validate = False)
