from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

from continuoussets.unary_operations.convert import Convert
from continuoussets.unary_operations.represents import Represents

import continuoussets.convexsets.interval as interval
import continuoussets.convexsets.zonotope as zonotope
import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry
from continuoussets.utils.exceptions import UnboundedSetError, ExactEvaluationImpossibleError

if __name__ == '__main__':
    print('This is the CartesianProduct class.')


@StrategyRegistry
class CartesianProduct(IBinaryOperation):

    # does the order of operands matter?
    ordered = True

    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):
        super().__init__(S1, S2, mode = mode)

        # read out concrete implementation
        self = self.select_binary_strategy()
        
    def __call__(self) -> 'IConvexSet':
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@CartesianProduct.register_strategy(SetPair('ndarray', 'ndarray'))
def _cartesian_product_point_point(s1, s2, mode) -> np.ndarray:
    return np.hstack(s1, s2)


@CartesianProduct.register_strategy(SetPair('ndarray', 'Interval'))
def _cartesian_product_point_interval(s1, I2, mode) -> interval.Interval:
    return interval.Interval(lb = np.hstack((s1, I2.lb)),
                             ub = np.hstack((s1, I2.ub)),
                             validate = False)


@CartesianProduct.register_strategy(SetPair('ndarray', 'Zonotope'))
def _cartesian_product_point_zonotope(s1, Z2, mode) -> zonotope.Zonotope:
    return zonotope.Zonotope(c = np.hstack((s1, Z2.c)),
                             G = np.hstack((np.zeros((s1.size, Z2.number_generators())), Z2.G)),
                             validate = False)


@CartesianProduct.register_strategy(SetPair('ndarray', 'VPolytope'))
def _cartesian_product_point_vpolytope(s1, VP2, mode) -> vpolytope.VPolytope:
    VP1 = Convert(s1, 'VPolytope', mode = 'exact')()
    return _cartesian_product_vpolytope_vpolytope(VP1, VP2, mode = mode)


@CartesianProduct.register_strategy(SetPair('ndarray', 'HPolyhedron'))
def _cartesian_product_point_hpolyhedron(s1, HP2, mode) -> hpolyhedron.HPolyhedron:
    HP1 = Convert(s1, 'VPolytope', mode = 'exact')()
    return _cartesian_product_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)


@CartesianProduct.register_strategy(SetPair('Interval', 'ndarray'))
def _cartesian_product_interval_point(I1, s2, mode) -> interval.Interval:
    return interval.Interval(lb = np.hstack((I1.lb, s2)),
                             ub = np.hstack((I1.ub, s2)),
                             validate = False)


@CartesianProduct.register_strategy(SetPair('Interval', 'Interval'))
def _cartesian_product_interval_interval(I1, I2, mode) -> interval.Interval:
    return interval.Interval(lb = np.hstack((I1.lb, I2.lb)),
                             ub = np.hstack((I1.ub, I2.ub)),
                             validate = False)


@CartesianProduct.register_strategy((SetPair('Interval', 'Zonotope'),
                                     SetPair('Interval', 'VPolytope'),
                                     SetPair('Interval', 'HPolyhedron')))
def _cartesian_product_interval_zonotope(I1, S2, mode) -> interval.Interval:
    try:
        I2 = Convert(S2, 'Interval', mode = mode)()
    except (UnboundedSetError):
        HP1 = Convert(I1, 'HPolyhedron', mode = 'exact')()
        return CartesianProduct(HP1, S2, mode = mode)()
    except (ExactEvaluationImpossibleError, NotImplementedError):  # mode = 'exact
        S1 = Convert(I1, type(S2).__name__, mode = mode)()
        return CartesianProduct(S1, S2, mode = mode)()
    
    return _cartesian_product_interval_interval(I1, I2, mode = mode)


@CartesianProduct.register_strategy(SetPair('Zonotope', 'ndarray'))
def _cartesian_product_zonotope_point(Z1, s2, mode) -> zonotope.Zonotope:
    return zonotope.Zonotope(c = np.hstack((Z1.c, s2)),
                             G = np.hstack((Z1.G, np.zeros((Z1.number_generators(), s2.size)))),
                             validate = False)


@CartesianProduct.register_strategy(SetPair('Zonotope', 'Interval'))
def _cartesian_product_zonotope_interval(Z1, I2, mode) -> zonotope.Zonotope:
    Z2 = Convert(I2, 'Zonotope', mode = mode)()
    return _cartesian_product_zonotope_zonotope(Z1, Z2, mode = mode)


@CartesianProduct.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _cartesian_product_zonotope_zonotope(Z1, Z2, mode) -> zonotope.Zonotope:
    n1, m1 = Z1.dimension, Z1.number_generators()
    n2, m2 = Z2.dimension, Z2.number_generators()

    # concatenate centers, block-concatenate generator matrices
    center = np.hstack((Z1.c, Z2.c))
    generators = np.vstack((np.hstack((Z1.G, np.zeros((m1, n2)))),
                            np.hstack((np.zeros((m2, n1)), Z2.G))))

    return zonotope.Zonotope(c = center, G = generators, validate = False)


@CartesianProduct.register_strategy(SetPair('Zonotope', 'VPolytope'))
def _cartesian_product_zonotope_vpolytope(Z1, VP2, mode) -> zonotope.Zonotope:
    if VP2.dimension == 1:
        Z2 = Convert(VP2, 'Zonotope', mode = mode)()
        return _cartesian_product_zonotope_zonotope(Z1, Z2, mode = mode)
    elif Represents(VP2, 'ndarray', rtol = 1e-12, atol = 1e-12)():
        return _cartesian_product_zonotope_point(Z1, VP2.center(), mode = mode)
    
    VP1 = Convert(Z1, 'VPolytope', mode = mode)()
    return _cartesian_product_vpolytope_vpolytope(VP1, VP2, mode = mode)


@CartesianProduct.register_strategy(SetPair('Zonotope', 'HPolyhedron'))
def _cartesian_product_zonotope_hpolyhedron(Z1, HP2, mode) -> zonotope.Zonotope:
    if HP2.dimension == 1 and not HP2.empty() and HP2.bounded():
        Z2 = Convert(HP2, 'Zonotope', mode = mode)()
        return _cartesian_product_zonotope_zonotope(Z1, Z2, mode = mode)
    elif Represents(HP2, 'ndarray', rtol = 1e-12, atol = 1e-12)():
        return _cartesian_product_zonotope_point(Z1, HP2.center(), mode = mode)
    
    HP1 = Convert(Z1, 'HPolyhedron', mode = mode)()
    return _cartesian_product_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)


@CartesianProduct.register_strategy((SetPair('VPolytope', 'ndarray'),
                                     SetPair('VPolytope', 'Interval'),
                                     SetPair('VPolytope', 'Zonotope')))
def _cartesian_product_vpolytope_other(VP1, S2, mode) -> vpolytope.VPolytope:
    VP2 = Convert(S2, 'VPolytope', mode = mode)()
    return _cartesian_product_vpolytope_vpolytope(VP1, VP2, mode = mode)


@CartesianProduct.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _cartesian_product_vpolytope_vpolytope(VP1, VP2, mode) -> vpolytope.VPolytope:
    # all potential combinations of vertices
    V_product = np.hstack((np.tile(VP1.V, (VP2.number_vertices(), 1)),
                           np.repeat(VP2.V, VP1.number_vertices(), axis = 0)))
    return vpolytope.VPolytope(V = V_product, validate = False)


@CartesianProduct.register_strategy(SetPair('VPolytope', 'HPolyhedron'))
def _cartesian_product_vpolytope_interval(VP1, HP2, mode) -> vpolytope.VPolytope:
    try:
        VP2 = Convert(HP2, 'VPolytope', mode = mode)()
    except (UnboundedSetError):
        # represent result as HPolyhedron
        HP1 = Convert(VP1, 'HPolyhedron', mode = mode)()
        return _cartesian_product_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)
    return _cartesian_product_vpolytope_vpolytope(VP1, VP2, mode = mode)


@CartesianProduct.register_strategy((SetPair('HPolyhedron', 'ndarray'),
                                     SetPair('HPolyhedron', 'Interval'),
                                     SetPair('HPolyhedron', 'Zonotope'),
                                     SetPair('HPolyhedron', 'VPolytope')))
def _cartesian_product_hpolyhedron_other(HP1, S2, mode) -> hpolyhedron.HPolyhedron:
    HP2 = Convert(S2, 'HPolyhedron', mode = mode)()
    return _cartesian_product_hpolyhedron_hpolyhedron(HP1, HP2, mode = mode)


@CartesianProduct.register_strategy(SetPair('HPolyhedron', 'HPolyhedron'))
def _cartesian_product_hpolyhedron_hpolyhedron(HP1, HP2, mode) -> hpolyhedron.HPolyhedron:
    # block-concatenation of constraint matrices, stack constraint offsets
    n1 = HP1.dimension
    n2 = HP2.dimension
    h1 = HP1.number_constraints()
    h2 = HP2.number_constraints()
    A_new = np.vstack((np.hstack((HP1.A, np.zeros((h1, n2)))),
                       np.hstack((np.zeros((h2, n1)), HP2.A))))
    b_new = np.hstack((HP1.b, HP2.b))

    return hpolyhedron.HPolyhedron(A = A_new, b = b_new, validate = False)
