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

from continuoussets.utils.auxiliary import SetPair

# ! we need ...


# class for all containment checks
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
        
    def __call__(self) -> bool:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@MinkowskiDifference.register_strategy(SetPair('ndarray', 'ndarray'))
def _minkowski_difference_point_point(s1, s2, mode) -> np.ndarray:
    return np.hstack(s1, s2)


@MinkowskiDifference.register_strategy(SetPair('ndarray', 'Interval'))
def _minkowski_difference_point_interval(s1, I2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('ndarray', 'Zonotope'))
def _minkowski_difference_point_zonotope(s1, Z2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('ndarray', 'VPolytope'))
def _minkowski_difference_point_vpolytope(s1, VP2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('ndarray', 'HPolyhedron'))
def _minkowski_difference_point_hpolyhedron(s1, HP2, mode) -> hpolyhedron.HPolyhedron:
    pass


@MinkowskiDifference.register_strategy(SetPair('Interval', 'ndarray'))
def _minkowski_difference_interval_point(I1, s2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('Interval', 'Interval'))
def _minkowski_difference_interval_interval(I1, I2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('Interval', 'Zonotope'))
def _minkowski_difference_interval_zonotope(I1, Z2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('Interval', 'VPolytope'))
def _minkowski_difference_interval_vpolytope(I1, VP2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('Interval', 'HPolyhedron'))
def _minkowski_difference_interval_hpolyhedron(I1, HP2, mode) -> interval.Interval:
    pass


@MinkowskiDifference.register_strategy(SetPair('Zonotope', 'ndarray'))
def _minkowski_difference_zonotope_point(Z1, s2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('Zonotope', 'Interval'))
def _minkowski_difference_zonotope_interval(Z1, I2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _minkowski_difference_zonotope_zonotope(Z1, Z2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('Zonotope', 'VPolytope'))
def _minkowski_difference_zonotope_vpolytope(Z1, VP2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('Zonotope', 'HPolyhedron'))
def _minkowski_difference_zonotope_hpolyhedron(Z1, HP2, mode) -> zonotope.Zonotope:
    pass


@MinkowskiDifference.register_strategy(SetPair('VPolytope', 'ndarray'))
def _minkowski_difference_vpolytope_point(VP1, s2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('VPolytope', 'Interval'))
def _minkowski_difference_vpolytope_interval(VP1, I2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _minkowski_difference_vpolytope_zonotope(VP1, Z2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _minkowski_difference_vpolytope_vpolytope(VP1, VP2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('VPolytope', 'HPolyhedron'))
def _minkowski_difference_vpolytope_hpolyhedron(VP1, HP2, mode) -> vpolytope.VPolytope:
    pass


@MinkowskiDifference.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _minkowski_difference_hpolyhedron_point(HP1, s2, mode) -> hpolyhedron.HPolyhedron:
    pass


@MinkowskiDifference.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _minkowski_difference_hpolyhedron_interval(HP1, I2, mode) -> hpolyhedron.HPolyhedron:
    pass


@MinkowskiDifference.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _minkowski_difference_hpolyhedron_zonotope(HP1, Z2, mode) -> hpolyhedron.HPolyhedron:
    pass


@MinkowskiDifference.register_strategy(SetPair('HPolyhedron', 'VPolytope'))
def _minkowski_difference_hpolyhedron_vpolytope(HP1, VP2, mode) -> hpolyhedron.HPolyhedron:
    pass


@MinkowskiDifference.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _minkowski_difference_hpolyhedron_hpolyhedron(HP1, HP2, mode) -> hpolyhedron.HPolyhedron:
    pass
