from __future__ import annotations

from typing import TYPE_CHECKING, Union, Dict, Tuple, Callable
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation, SetPair
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


# class for all containment checks
class Contains(IBinaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is relevant
    ordered_operation = True
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 rtol: float, atol: float):

        # call superclass constructor
        super().__init__(S1, S2, rtol = rtol, atol = atol)

        # get concrete implementation function
        strategy_key = self.get_strategy_key()
        self.func = Contains.strategies.get(strategy_key, None)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError
        
    def __call__(self) -> bool:
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


# different strategies for containment check
@Contains.register_strategy(SetPair('Interval', 'ndarray'))
def _contains_interval_point(S1, S2, rtol, atol) -> bool:
    return True


@Contains.register_strategy(SetPair('Interval', 'Interval'))
def _contains_interval_interval(S1, S2, rtol, atol) -> bool:
    return True


@Contains.register_strategy(SetPair('Interval', 'Zonotope'))
def _contains_interval_zonotope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Interval', 'VPolytope'))
def _contains_interval_vpolytope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Interval', 'HPolyhedron'))
def _contains_interval_hpolyhedron(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Zonotope', 'ndarray'))
def _contains_zonotope_point(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Zonotope', 'Interval'))
def _contains_zonotope_interval(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _contains_zonotope_zonotope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Zonotope', 'VPolytope'))
def _contains_zonotope_vpolytope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('Zonotope', 'HPolyhedron'))
def _contains_zonotope_hpolyhedron(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('VPolytope', 'ndarray'))
def _contains_vpolytope_point(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('VPolytope', 'Interval'))
def _contains_vpolytope_interval(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _contains_vpolytope_zonotope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _contains_vpolytope_vpolytope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('VPolytope', 'HPolyhedron'))
def _contains_vpolytope_hpolyhedron(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _contains_hpolyhedron_point(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _contains_hpolyhedron_interval(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _contains_hpolyhedron_zonotope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('HPolyhedron', 'VPolytope'))
def _contains_hpolyhedron_vpolytope(S1, S2, rtol, atol) -> bool:
    pass


@Contains.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _contains_hpolyhedron_hpolyhedron(S1, S2, rtol, atol) -> bool:
    pass
