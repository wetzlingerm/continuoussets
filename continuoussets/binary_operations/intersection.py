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

from continuoussets.binary_operations.contains import Contains
from continuoussets.unary_operations.convert import Convert

from continuoussets.utils.auxiliary import SetPair
from continuoussets.utils.exceptions import EmptySetError

if __name__ == '__main__':
    print('This is the Intersection class.')


class Intersection(IBinaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is not relevant
    ordered_operation = False
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):

        # call superclass constructor
        super().__init__(S1, S2, mode)

        # get concrete implementation function
        strategy_key = self.get_strategy_key()
        self.func = Intersection.strategies.get(strategy_key, None)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError
        
    def __call__(self) -> 'IConvexSet':
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@Intersection.register_strategy(SetPair('ndarray', 'ndarray'))
def _intersection_point_point(s1, s2, mode) -> np.ndarray:
    if np.allclose(s1, s2, rtol = 1e-12, atol = 1e-12):
        return s1
    return EmptySetError


@Intersection.register_strategy((SetPair('Interval', 'ndarray'),
                                 SetPair('Zonotope', 'ndarray'),
                                 SetPair('VPolytope', 'ndarray'),
                                 SetPair('HPolyhedron', 'ndarray')))
def _intersection_any_point(S1, s2, mode) -> np.ndarray:
    if Contains(S1, s2, rtol = 1e-12, atol = 1e-12)():
        return s2
    return EmptySetError


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
    HP1 = Convert(S1, 'HPolyhedron', mode = mode)
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
