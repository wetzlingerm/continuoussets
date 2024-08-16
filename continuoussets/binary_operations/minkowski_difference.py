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
from continuoussets.unary_operations.represents import Represents

from continuoussets.utils.auxiliary import SetPair
from continuoussets.utils.exceptions import UnboundedSetError, EmptySetError

if __name__ == '__main__':
    print('This is the MinkowskiDifference class.')


class MinkowskiDifference(IBinaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is relevant
    ordered_operation = True
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):

        # call superclass constructor
        super().__init__(S1, S2, mode)

        # get concrete implementation function
        strategy_key = self.get_strategy_key()
        self.func = MinkowskiDifference.strategies.get(strategy_key, None)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError
        
    def __call__(self) -> Union[np.ndarray, 'IConvexSet']:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@MinkowskiDifference.register_strategy((SetPair('ndarray', 'ndarray'),
                                        SetPair('Interval', 'ndarray'),
                                        SetPair('Zonotope', 'ndarray'),
                                        SetPair('VPolytope', 'ndarray'),
                                        SetPair('HPolyhedron', 'ndarray')))
def _minkowski_difference_any_point(S1, s2, mode) -> Union[np.ndarray, 'IConvexSet']:
    return S1 - s2


@MinkowskiDifference.register_strategy((SetPair('ndarray', 'Interval'),
                                        SetPair('ndarray', 'Zonotope'),
                                        SetPair('ndarray', 'VPolytope'),
                                        SetPair('ndarray', 'HPolyhedron')))
def _minkowski_difference_point_interval(s1, S2, mode) -> np.ndarray:
    if not Represents(S2, 'ndarray', rtol = 1e-12, atol = 1e-12)():
        raise EmptySetError
    return s1 - S2.center()


@MinkowskiDifference.register_strategy(SetPair('Interval', 'Interval'))
def _minkowski_difference_interval_interval(I1, S2, mode) -> interval.Interval:
    S2_diameter = S2.diameter()
    if np.any(I1.diameter() < S2_diameter):
        raise EmptySetError

    other_center = S2.center()
    return interval.Interval(lb = I1.lb - other_center + 0.5*S2_diameter,
                             ub = I1.ub - other_center - 0.5*S2_diameter, validate = False)


@MinkowskiDifference.register_strategy((SetPair('Interval', 'Zonotope'),
                                        SetPair('Interval', 'VPolytope'),
                                        SetPair('Interval', 'HPolyhedron')))
def _minkowski_difference_interval_other(I1, S2, mode) -> interval.Interval:
    try:
        I2 = Convert(S2, 'Interval', mode = 'outer')()
    except (UnboundedSetError):
        raise EmptySetError

    return _minkowski_difference_interval_interval(I1, I2)


@MinkowskiDifference.register_strategy((SetPair('Zonotope', 'Interval'),
                                        SetPair('Zonotope', 'Zonotope'),
                                        SetPair('Zonotope', 'VPolytope'),
                                        SetPair('Zonotope', 'HPolyhedron')))
def _minkowski_difference_zonotope_other(Z1, S2, mode) -> zonotope.Zonotope:
    raise NotImplementedError


@MinkowskiDifference.register_strategy((SetPair('VPolytope', 'Interval'),
                                        SetPair('VPolytope', 'Zonotope'),
                                        SetPair('VPolytope', 'VPolytope'),
                                        SetPair('VPolytope', 'HPolyhedron')))
def _minkowski_difference_vpolytope_other(VP1, S2, mode) -> vpolytope.VPolytope:
    if not Represents(S2, 'ndarray', rtol = 1e-12, atol = 1e-12)():
        raise NotImplementedError
    return VP1 - S2.center()
    

@MinkowskiDifference.register_strategy((SetPair('HPolyhedron', 'Interval'),
                                        SetPair('HPolyhedron', 'Zonotope'),
                                        SetPair('HPolyhedron', 'VPolytope'),
                                        SetPair('HPolyhedron', 'HPolyhedron')))
def _minkowski_difference_hpolyhedron_other(HP1, S2, mode) -> hpolyhedron.HPolyhedron:
    b_new = HP1.b.copy()
    for i in range(HP1.number_constraints()):
        b_new[i] -= S2.support_function(HP1.A[i])[0]

    return hpolyhedron.HPolyhedron(A = HP1.A.copy(), b = b_new)
