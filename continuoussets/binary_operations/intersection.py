from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

import continuoussets.convexsets.interval as interval
# import continuoussets.convexsets.zonotope as zonotope
# import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.binary_operations.contains import Contains
from continuoussets.unary_operations.convert import Convert

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry
from continuoussets.utils.exceptions import EmptySetError

if __name__ == '__main__':
    print('This is the Intersection class.')


@StrategyRegistry
class Intersection(IBinaryOperation):

    # does the order of operands matter?
    ordered = False
    
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):
        super().__init__(S1, S2, mode = mode)

        # read out concrete implementation
        self = self.select_binary_strategy()
        
    def __call__(self) -> 'IConvexSet':
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@Intersection.register_strategy(SetPair('ndarray', 'ndarray'))
def _intersection_point_point(s1, s2, mode) -> np.ndarray:
    if not np.allclose(s1, s2, rtol = 1e-12, atol = 1e-12):
        raise EmptySetError
    return s1


@Intersection.register_strategy((SetPair('Interval', 'ndarray'),
                                 SetPair('Zonotope', 'ndarray'),
                                 SetPair('VPolytope', 'ndarray'),
                                 SetPair('HPolyhedron', 'ndarray')))
def _intersection_any_point(S1, s2, mode) -> np.ndarray:
    if not Contains(S1, s2, rtol = 1e-12, atol = 1e-12)():
        raise EmptySetError
    return s2


@Intersection.register_strategy(SetPair('Interval', 'Interval'))
def _intersection_interval_interval(I1, I2, mode) -> interval.Interval:
    lower = np.maximum(I1.lb, I2.lb)
    upper = np.minimum(I1.ub, I2.ub)
    if np.any(lower > upper):
        raise EmptySetError
    return interval.Interval(lb = lower, ub = upper, validate = False)


@Intersection.register_strategy((SetPair('Zonotope', 'Interval'),
                                 SetPair('Zonotope', 'Zonotope'),
                                 SetPair('VPolytope', 'Interval'),
                                 SetPair('VPolytope', 'Zonotope'),
                                 SetPair('VPolytope', 'VPolytope')))
def _intersection_other_other(S1, S2, mode) -> 'IConvexSet':
    if Contains(S1, S2, rtol = 1e-12, atol = 1e-12)():
        return S2.copy()
    if Contains(S2, S1, rtol = 1e-12, atol = 1e-12)():
        return S1.copy()
    HP1 = Convert(S1, 'HPolyhedron', mode = mode)()
    return _intersection_hpolyhedron_other(HP1, S2, mode = mode)


@Intersection.register_strategy((SetPair('HPolyhedron', 'Interval'),
                                 SetPair('HPolyhedron', 'Zonotope'),
                                 SetPair('HPolyhedron', 'VPolytope')))
def _intersection_hpolyhedron_other(HP1, S2, mode) -> 'IConvexSet':
    if Contains(HP1, S2, rtol = 1e-12, atol = 1e-12)():
        return S2.copy()
    
    HP2 = Convert(S2, 'HPolyhedron', mode = mode)()
    return _intersection_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)


@Intersection.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _intersection_hpolyhedron_hpolyhedron(HP1, HP2, mode) -> hpolyhedron.HPolyhedron:
    return hpolyhedron.HPolyhedron(A = np.vstack((HP1.A, HP2.A)),
                                   b = np.hstack((HP1.b, HP2.b)),
                                   validate = False)
