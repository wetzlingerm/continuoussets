from __future__ import annotations

from typing import Union, TYPE_CHECKING
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry
from continuoussets.utils.comparison import compare_matrices

from continuoussets.unary_operations.convert import Convert
from continuoussets.binary_operations.contains import Contains

if __name__ == '__main__':
    print('This is the Equals class.')


@StrategyRegistry
class Equals(IBinaryOperation):

    # does the order of operands matter?
    ordered = False
    
    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 rtol: float, atol: float):
        super().__init__(S1, S2, rtol = rtol, atol = atol)

        # read out concrete implementation
        self = self.select_binary_strategy()

    def __call__(self) -> bool:
        # in all cases: dimensions need to match
        # todo: write this smarter... maybe re-use dimension check...
        if isinstance(self.first_operand, np.ndarray):
            n1 = self.first_operand.shape[0]
        else:
            n1 = self.first_operand.dimension
        if isinstance(self.second_operand, np.ndarray):
            n2 = self.second_operand.shape[0]
        else:
            n2 = self.second_operand.dimension
        if n1 != n2:
            return False
        
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@Equals.register_strategy(SetPair('ndarray', 'ndarray'))
def _equals_point_point(s1, s2, rtol, atol) -> bool:
    return np.allclose(s1, s2, rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('Interval', 'ndarray'))
def _equals_interval_point(I1, s2, rtol, atol) -> bool:
    return _equals_interval_interval(I1, Convert(s2, 'Interval', mode = 'exact')(), rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('Interval', 'Interval'))
def _equals_interval_interval(I1, I2, rtol, atol) -> bool:
    return np.allclose(I1.lb, I2.lb, rtol = rtol, atol = atol) and \
           np.allclose(I1.ub, I2.ub, rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('Zonotope', 'ndarray'))
def _equals_zonotope_point(Z1, s2, rtol, atol) -> bool:
    return (np.allclose(Z1.c, s2, rtol = rtol, atol = atol)
            and Z1.represents('Point', rtol = rtol, atol = atol))


@Equals.register_strategy(SetPair('Zonotope', 'Interval'))
def _equals_zonotope_interval(Z1, I2, rtol, atol) -> bool:
    return _equals_zonotope_zonotope(Z1, Convert(I2, 'Zonotope', mode = 'exact')(), rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _equals_zonotope_zonotope(Z1, Z2, rtol, atol) -> bool:
    if not np.allclose(Z1.c, Z2.c, rtol = rtol, atol = atol):
        return False
    return compare_matrices(Z1.compact().G, Z2.compact().G,
                            rtol = rtol, atol = atol,
                            remove_zeros = True, check_negation = True)


@Equals.register_strategy((SetPair('VPolytope', 'ndarray'),
                           SetPair('VPolytope', 'Interval'),
                           SetPair('VPolytope', 'Zonotope')))
def _equals_vpolytope_other(VP1, S2, rtol, atol) -> bool:
    S2 = Convert(S2, 'VPolytope', mode = 'exact')()
    VP1, S2 = VP1.compact(), S2.compact()
    return compare_matrices(VP1.V, S2.V, rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _equals_vpolytope_vpolytope(VP1, VP2, rtol, atol) -> bool:
    VP1, VP2 = VP1.compact(), VP2.compact()
    return compare_matrices(VP1.V, VP2.V, rtol = rtol, atol = atol)


@Equals.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _equals_hpolyhedron_point(HP1, s2, rtol, atol) -> bool:
    if not Contains(HP1, s2, rtol = rtol, atol = atol)():
        return False
    HP2 = Convert(s2, 'HPolyhedron', mode = 'exact')()
    return Contains(HP2, HP1, rtol = rtol, atol = atol)()


@Equals.register_strategy((SetPair('HPolyhedron', 'Interval'),
                           SetPair('HPolyhedron', 'Zonotope'),
                           SetPair('HPolyhedron', 'VPolytope')))
def _equals_hpolyhedron_other(HP1, S2, rtol, atol) -> bool:
    HP2 = Convert(S2, 'HPolyhedron', mode = 'exact')()
    return _equals_hpolyhedron_hpolyhedron(HP1, HP2, rtol, atol)


@Equals.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _equals_hpolyhedron_hpolyhedron(HP1, HP2, rtol, atol) -> bool:
    return (Contains(HP1, HP2, rtol = rtol, atol = atol)()
            and Contains(HP2, HP1, rtol = rtol, atol = atol)())
