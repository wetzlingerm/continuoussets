from __future__ import annotations

from typing import Union, Dict, Tuple, Callable, TYPE_CHECKING
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation, SetPair
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


# class for all equality checks
class Equals(IBinaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is not relevant
    ordered_operation = False
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 rtol: float, atol: float):

        # call superclass constructor
        super().__init__(S1, S2, rtol = rtol, atol = atol)

        # get concrete implementation function
        strategy_key: SetPair = self.get_strategy_key()
        # read out function
        self.func: Callable = Equals.strategy.get(strategy_key)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError

    # evaluate the set equality
    def __call__(self) -> bool:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@Equals.register_strategy(SetPair('Interval', 'ndarray'))
def _equals_interval_point(I1, s2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('Interval', 'Interval'))
def _equals_interval_interval(I1, I2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('Zonotope', 'ndarray'))
def _equals_zonotope_point(Z1, s2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('Zonotope', 'Interval'))
def _equals_zonotope_interval(Z1, I2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _equals_zonotope_zonotope(Z1, Z2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('VPolytope', 'ndarray'))
def _equals_vpolytope_point(VP1, s2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('VPolytope', 'Interval'))
def _equals_vpolytope_interval(VP1, I2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _equals_vpolytope_zonotope(VP1, Z2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _equals_vpolytope_vpolytope(VP1, VP2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _equals_hpolyhedron_point(HP1, s2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _equals_hpolyhedron_interval(HP1, I2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _equals_hpolyhedron_zonotope(HP1, Z2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('HPolyhedron', 'VPolytope'))
def _equals_hpolyhedron_vpolytope(HP1, VP2, rtol, atol) -> bool:
    pass


@Equals.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _equals_hpolyhedron_hpolyhedron(HP1, HP2, rtol, atol) -> bool:
    pass
