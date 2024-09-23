from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np

from continuoussets.binary_operations.interface_binary_operation import IBinaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

import continuoussets.convexsets.interval as interval
import continuoussets.convexsets.zonotope as zonotope
import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.binary_operations.contains import Contains
from continuoussets.unary_operations.represents import Represents
from continuoussets.unary_operations.convert import Convert

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry
from continuoussets.utils.exceptions import UnboundedSetError

if __name__ == '__main__':
    print('This is the ConvexHull class.')


@StrategyRegistry
class ConvexHull(IBinaryOperation):

    # does the order of operands matter?
    ordered = False

    def __init__(self, S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
                 mode: str):
        super().__init__(S1, S2, mode = mode)

        # read out concrete implementation
        self = self.select_binary_strategy()

    def __call__(self) -> 'IConvexSet':
        return self.func(self.first_operand, self.second_operand, **self.kwargs)


@ConvexHull.register_strategy(SetPair('ndarray', 'ndarray'))
def _convex_hull_point_point(s1, s2, mode) -> Union[np.ndarray, zonotope.Zonotope]:
    if np.allclose(s1, s2, rtol = 1e-12, atol = 1e-12):
        return s1

    center = (s1 + s2) / 2.
    generator = s2 - center
    return zonotope.Zonotope(c = center, G = generator, validate = False)


@ConvexHull.register_strategy(SetPair('Interval', 'ndarray'))
def _convex_hull_interval_point(I1, s2, mode) -> interval.Interval:
    if mode == 'inner':
        return I1.copy()
    
    if mode == 'exact':
        if Contains(I1, s2, rtol = 1e-12, atol = 1e-12)():
            return I1.copy()
        VP1 = Convert(I1, 'VPolytope', mode = 'exact')()
        return _convex_hull_vpolytope_other(VP1, s2, mode = 'exact')
    
    if mode == 'outer':
        I2 = Convert(s2, 'Interval', mode = 'exact')()
        return _convex_hull_interval_interval(I1, I2, mode = mode)


@ConvexHull.register_strategy(SetPair('Interval', 'Interval'))
def _convex_hull_interval_interval(I1, I2, mode) -> interval.Interval:
    if mode == 'inner':
        if Contains(I1, I2, rtol = 1e-12, atol = 1e-12)():
            return I1.copy()
        elif Contains(I2, I1, rtol = 1e-12, atol = 1e-12)():
            return I2.copy()
        raise NotImplementedError
    
    if mode == 'exact':
        if Contains(I1, I2, rtol = 1e-12, atol = 1e-12)():
            return I1.copy()
        elif Contains(I2, I1, rtol = 1e-12, atol = 1e-12)():
            return I2.copy()
        VP1 = Convert(I1, 'VPolytope', mode = 'exact')()
        return _convex_hull_vpolytope_other(VP1, I2, mode = 'exact')

    if mode == 'outer':
        return interval.Interval(lb = np.minimum(I1.lb, I2.lb),
                                 ub = np.maximum(I1.ub, I2.ub), validate=False)


@ConvexHull.register_strategy((SetPair('Zonotope', 'ndarray'),
                               SetPair('Zonotope', 'Interval')))
def _convex_hull_zonotope_other(Z1, S2, mode) -> zonotope.Zonotope:
    Z2 = Convert(S2, 'Zonotope', mode = mode)()
    return _convex_hull_zonotope_zonotope(Z1, Z2, mode = mode)


@ConvexHull.register_strategy(SetPair('Zonotope', 'Zonotope'))
def _convex_hull_zonotope_zonotope(Z1, Z2, mode) -> zonotope.Zonotope:
    if mode == 'inner':
        # pick one of the zonotopes...
        return Z1.copy()

    elif mode == 'exact':
        if (Represents(Z1, 'ndarray', rtol = 1e-12, atol = 1e-12)()
                and Represents(Z1, 'ndarray', rtol = 1e-12, atol = 1e-12)()):
            # here, outer approximation algorithm returns the exact solution
            return _convex_hull_zonotope_zonotope(Z1, Z2, mode = 'outer')
        if Contains(Z1, Z2, rtol = 1e-12, atol = 1e-12)():
            return Z1.copy()
        if Contains(Z2, Z1, rtol = 1e-12, atol = 1e-12)():
            return Z2.copy()
        VP1 = Convert(Z1, 'VPolytope', mode = 'exact')()
        return _convex_hull_vpolytope_other(VP1, Z2, mode = 'exact')

    if mode == 'outer':
        center = 0.5 * (Z1.c + Z2.c)
        generator_center = 0.5 * (Z1.c - Z2.c)

        m1, m2 = Z1.number_generators(), Z2.number_generators()

        if m1 >= m2:
            generators = np.vstack((generator_center,
                                    0.5 * (Z1.G[:m2, :] + Z2.G),
                                    0.5 * (Z1.G[:m2, :] - Z2.G),
                                    Z1.G[m2:, :]))
        else:
            generators = np.vstack((generator_center,
                                    0.5 * (Z1.G + Z2.G[:m1, :]),
                                    0.5 * (Z1.G - Z2.G[:m1, :]),
                                    Z2.G[m1:, :]))

        return zonotope.Zonotope(c = center, G = generators, validate = False)


@ConvexHull.register_strategy((SetPair('VPolytope', 'ndarray'),
                               SetPair('VPolytope', 'Interval'),
                               SetPair('VPolytope', 'Zonotope')))
def _convex_hull_vpolytope_other(VP1, S2, mode) -> vpolytope.VPolytope:
    VP2 = Convert(S2, 'VPolytope', mode = mode)()
    return _convex_hull_vpolytope_vpolytope(VP1, VP2, mode = mode)


@ConvexHull.register_strategy(SetPair('VPolytope', 'VPolytope'))
def _convex_hull_vpolytope_vpolytope(VP1, VP2, mode) -> vpolytope.VPolytope:
    V_all = np.vstack((VP1.V, VP2.V))
    return vpolytope.VPolytope(V = V_all, validate = False)


@ConvexHull.register_strategy((SetPair('HPolyhedron', 'ndarray'),
                               SetPair('HPolyhedron', 'Interval'),
                               SetPair('HPolyhedron', 'Zonotope'),
                               SetPair('HPolyhedron', 'VPolytope'),
                               SetPair('HPolyhedron', 'HPolyhedron')))
def _convex_hull_hpolyhedron_other(HP1, S2, mode) -> hpolyhedron.HPolyhedron:
    if mode == 'inner':
        return HP1.copy()
    
    if mode == 'exact':
        if Contains(HP1, S2, rtol = 1e-12, atol = 1e-12)():
            return HP1.copy()
        if Contains(S2, HP1, rtol = 1e-12, atol = 1e-12)():
            return S2.copy()
        try:
            VP1 = Convert(HP1, 'VPolytope', mode = 'exact')()
        except (UnboundedSetError):
            raise NotImplementedError
        return _convex_hull_vpolytope_other(VP1, S2, mode = mode)
    
    if mode == 'outer':
        def _support_value(S, direction):
            if isinstance(S, np.ndarray):
                return np.dot(S, direction)
            return S.support_function(direction)[0]
        
        h = HP1.number_constraints()
        
        # 'outer': compute support function of HP1+S2, take larger value, additional constraints from box
        A_new = np.vstack((HP1.A, np.eye(HP1.dimension()), -np.eye(HP1.dimension())))
        b_new = np.zeros(h + 2*HP1.dimension())

        # for the first constraints, we already have the value computed for the HPolyhedron
        for i in range(h):
            # compute support function value of S2 set
            value = _support_value(S2, HP1.A[i])
            b_new[i] = HP1.b[i] if HP1.b[i] > value else value

        # for the remaining constraints, we also have to evaluate the support function for the HPolyhedron
        for i in range(2*HP1.dimension()):
            value_polyhedron = HP1.support_function(A_new[h+i])[0]
            value = _support_value(S2, A_new[h+i])
            b_new[h+i] = value_polyhedron if value_polyhedron > value else value

        return hpolyhedron.HPolyhedron(A = A_new, b = b_new, validate = False)
