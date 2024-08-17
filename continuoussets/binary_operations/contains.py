from __future__ import annotations

from typing import TYPE_CHECKING, Union, Dict, Tuple, Callable
import numpy as np

from scipy.optimize import linprog

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

from continuoussets.utils.auxiliary import SetPair
from continuoussets.utils.exceptions import UnboundedSetError

from continuoussets.unary_operations.convert import Convert
from continuoussets.unary_operations.represents import Represents

if __name__ == '__main__':
    print('This is the Contains class.')


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
def _contains_interval_point(I1, s2, rtol, atol) -> bool:
    # todo: use tolerances
    return np.all(I1.lb <= s2) and np.all(I1.ub >= s2)


@Contains.register_strategy(SetPair('Interval', 'Interval'))
def _contains_interval_interval(I1, I2, rtol, atol) -> bool:
    # todo: use tolerances
    return np.all(I1.lb <= I2.lb) and np.all(I1.ub >= I2.ub)


@Contains.register_strategy((SetPair('Interval', 'Zonotope'),
                             SetPair('Interval', 'VPolytope'),
                             SetPair('Interval', 'HPolyhedron')))
def _contains_interval_other(I1, S2, rtol, atol) -> bool:
    # method that works for all convex sets as inner bodies
    try:
        S2 = Convert(S2, 'Interval', mode = 'outer')()
    except (UnboundedSetError):
        return False
    return _contains_interval_interval(I1, S2, rtol = rtol, atol = atol)


@Contains.register_strategy(SetPair('Zonotope', 'ndarray'))
def _contains_zonotope_point(Z1, s2, rtol, atol) -> bool:
    if Z1.number_generators() == 0:
        return np.allclose(Z1.c, s2, rtol = rtol, atol = atol)
    
    # shift zonotope and point by center of zonotope and check zonotope norm
    norm = (Z1 - Z1.c).zonotope_norm(s2 - Z1.c)
    return norm <= 1 or np.isclose(norm, 1., rtol = rtol, atol = atol)


@Contains.register_strategy((SetPair('Zonotope', 'Interval'),
                             SetPair('Zonotope', 'Zonotope'),
                             SetPair('Zonotope', 'VPolytope'),
                             SetPair('Zonotope', 'HPolyhedron')))
def _contains_zonotope_other(Z1, S2, rtol, atol) -> bool:
    if Represents(S2, 'ndarray', rtol = rtol, atol = atol)():
        return _contains_zonotope_point(Z1, S2.center(), rtol, atol)
    
    HP1 = Convert(Z1, 'HPolyhedron', mode = 'exact')()
    return Contains(HP1, S2, rtol = rtol, atol = atol)()


@Contains.register_strategy(SetPair('VPolytope', 'ndarray'))
def _contains_vpolytope_point(VP1, s2, rtol, atol) -> bool:
    # todo: use dual and integrate tolerances...
    # formulate containment as linear program: vector is contained if the LP is feasible
    c = np.zeros(VP1.number_vertices())
    A_eq = np.vstack((VP1.V.T, np.ones((1, VP1.number_vertices()))))
    b_eq = np.hstack((s2, 1))
    A_ub = -np.eye(VP1.number_vertices())
    b_ub = np.zeros((VP1.number_vertices(), 1))
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

    return res.success


@Contains.register_strategy((SetPair('VPolytope', 'Interval'),
                             SetPair('VPolytope', 'Zonotope'),
                             SetPair('VPolytope', 'VPolytope'),
                             SetPair('VPolytope', 'HPolyhedron')))
def _contains_vpolytope_other(VP1, S2, rtol, atol) -> bool:
    V = S2.vertices()
    for i in range(V.shape[0]):
        if not _contains_vpolytope_point(VP1, V[i], rtol = rtol, atol = atol):
            return False
    return True


@Contains.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _contains_hpolyhedron_point(HP1, s2, rtol, atol) -> bool:
    values = np.matmul(HP1.A, s2)
    # check absolute tolerance... only then check relative tolerance
    if not np.all(values <= HP1.b + atol):
        min_value = np.min((np.abs(values), np.abs(HP1.b)), axis = 0)
        if np.any(min_value == 0.) or np.any(np.abs(values - HP1.b) / min_value > rtol):
            return False
    return True


@Contains.register_strategy((SetPair('HPolyhedron', 'Interval'),
                             SetPair('HPolyhedron', 'Zonotope'),
                             SetPair('HPolyhedron', 'VPolytope'),
                             SetPair('HPolyhedron', 'HPolyhedron')))
def _contains_hpolyhedron_other(HP1, S2, rtol, atol) -> bool:
    for i in range(HP1.number_constraints()):
        # compute support function value of in-body along all normal vectors in A and compare to b
        value = S2.support_function(HP1.A[i])[0]
        # check absolute tolerance... only then check relative tolerance
        if value > HP1.b[i] + atol:
            min_value = np.min((np.abs(value), np.abs(HP1.b[i])))
            if (min_value == 0.) or (np.abs(value - HP1.b[i]) / min_value > rtol):
                return False
    
    return True
