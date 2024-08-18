from __future__ import annotations

from typing import Union, Callable, TYPE_CHECKING
from itertools import combinations
import numpy as np
from math import comb
from scipy.linalg import svd
from pypoman import compute_polytope_halfspaces

from continuoussets.unary_operations.interface_unary_operation import IUnaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

# required for instantiation (and type checking)
import continuoussets.convexsets.interval as interval
import continuoussets.convexsets.zonotope as zonotope
import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.unary_operations.represents import Represents

from continuoussets.utils.auxiliary import SetPair, StrategyRegistry, n_dim_cross_product
from continuoussets.utils.exceptions import ExactEvaluationImpossibleError, \
                                            UnboundedSetError, \
                                            EmptySetError

if __name__ == '__main__':
    print('This is the Convert class.')


@StrategyRegistry
class Convert(IUnaryOperation):
    
    def __init__(self, S: Union['IConvexSet', np.ndarray], set_class: str, *, mode: str):
        super().__init__(S, set_class, mode = mode)
        
        # read out concrete implementation
        self.func: Callable = Convert.select_strategy(self.strategy_key)

        if self.func is None:
            raise NotImplementedError

    def __call__(self) -> 'IConvexSet':
        return self.func(self.first_operand, **self.kwargs)


@Convert.register_strategy((SetPair('ndarray', 'ndarray'),
                            SetPair('Interval', 'Interval'),
                            SetPair('Zonotope', 'Zonotope'),
                            SetPair('VPolytope', 'VPolytope'),
                            SetPair('HPolyhedron', 'HPolyhedron')))
def _convert_any_self(S, mode) -> Union[np.ndarray, 'IConvexSet']:
    return S.copy()


@Convert.register_strategy(SetPair('ndarray', 'Interval'))
def _convert_point_interval(s, mode) -> interval.Interval:
    return interval.Interval(lb = s, ub = s, validate = False)


@Convert.register_strategy(SetPair('ndarray', 'Zonotope'))
def _convert_point_zonotope(s, mode) -> zonotope.Zonotope:
    return zonotope.Zonotope(c = s, validate = False)


@Convert.register_strategy(SetPair('ndarray', 'VPolytope'))
def _convert_point_vpolytope(s, mode) -> vpolytope.VPolytope:
    return vpolytope.VPolytope(V = s, validate = False)


@Convert.register_strategy(SetPair('ndarray', 'HPolyhedron'))
def _convert_point_hpolyhedron(s, mode) -> hpolyhedron.HPolyhedron:
    A = np.vstack((-np.ones(s.size), np.eye(s.size)))
    b = np.matmul(A, s)
    return hpolyhedron.HPolyhedron(A = A, b = b, validate = False)


@Convert.register_strategy((SetPair('Interval', 'ndarray'),
                            SetPair('Zonotope', 'ndarray'),
                            SetPair('VPolytope', 'ndarray'),
                            SetPair('HPolyhedron', 'ndarray')))
def _convert_interval_point(S, mode) -> np.ndarray:
    if mode == 'inner':
        return S.center()
    
    if mode == 'exact':
        if not Represents(S, 'ndarray', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        return S.center()

    if mode == 'outer':
        if not Represents(S, 'ndarray', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        return S.center()


@Convert.register_strategy(SetPair('Interval', 'Zonotope'))
def _convert_interval_zonotope(I, mode) -> zonotope.Zonotope:
    # ***same method for all three modes (exact)
    generators = np.diag(I.diameter())
    generators = 0.5*generators[~np.all(generators == 0, axis=0), :]
    return zonotope.Zonotope(c = I.center(), G = generators, validate = False)


@Convert.register_strategy((SetPair('Interval', 'VPolytope'),
                            SetPair('Zonotope', 'VPolytope'),
                            SetPair('HPolyhedron', 'VPolytope')))
def _convert_other_vpolytope(S, mode) -> vpolytope.VPolytope:
    try:
        VP = vpolytope.VPolytope(V = S.vertices(), validate = False)
    except (UnboundedSetError):
        raise ExactEvaluationImpossibleError
    return VP


@Convert.register_strategy(SetPair('Interval', 'HPolyhedron'))
def _convert_interval_hpolyhedron(I, mode) -> hpolyhedron.HPolyhedron:
    # ***same method for all three modes (exact)
    return hpolyhedron.HPolyhedron(A = np.vstack((np.eye(I.dimension), -np.eye(I.dimension))),
                                   b = np.hstack((I.ub, -I.lb)),
                                   validate = False)


@Convert.register_strategy(SetPair('Zonotope', 'Interval'))
def _convert_zonotope_interval(Z, mode) -> interval.Interval:
    if mode == 'inner':
        if not Represents(Z, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_zonotope_interval(Z, mode = 'outer')
    
    if mode == 'exact':
        if not Represents(Z, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        return _convert_zonotope_interval(Z, mode = 'outer')

    if mode == 'outer':
        radius = np.sum(np.abs(Z.G), axis = 0)
        return interval.Interval(lb = Z.c - radius, ub = Z.c + radius, validate = False)


@Convert.register_strategy(SetPair('Zonotope', 'HPolyhedron'))
def _convert_zonotope_hpolyhedron(Z, mode) -> hpolyhedron.HPolyhedron:
    # ***same method for all three modes (exact)
    # special instantiation if zonotope is a single point
    if Represents(Z, 'ndarray', rtol = 1e-12, atol = 1e-12)():
        return _convert_point_hpolyhedron(Z.center(), mode = 'exact')

    # conversion requires linearly independent generators
    Z = Z.compact()
    n_orig = Z.dimension

    if Z.degenerate():
        # shift by center and project onto affine hull
        (Z, M_proj, c) = Z.project_affine_hull()
    
    # pre-allocate constraint matrix and constraint offset
    n, m = Z.dimension, Z.number_generators()
    h = comb(m, n-1)
    A, b = np.zeros((2*h, n)), np.zeros(2*h)

    # we compute the n-dimensional cross product of all combinations of n-1 generators
    all_combinations = combinations(range(m), r = n-1)
    for row, combination in enumerate(all_combinations):
        cross_product = n_dim_cross_product(Z.G[list(combination)].T)
        A[row] = cross_product / np.linalg.norm(cross_product, ord = 2)
        A[row+h] = -A[row]
        delta = np.sum(np.abs(np.matmul(A[row], Z.G.T)))
        b[row] = np.matmul(A[row], Z.c) + delta
        b[row+h] = -np.matmul(A[row], Z.c) + delta

    # back-projection
    if n_orig > n:
        # additional constraints flattening other dimensions to 0
        A = np.block([[A, np.zeros((2*h, n_orig-n))],
                      [np.zeros((n_orig-n, n)), np.eye(n_orig-n)],
                      [np.zeros((n_orig-n, n)), -np.eye(n_orig-n)]])
        b = np.hstack((b, np.zeros(2*(n_orig-n))))
        # map constraint matrix of polytope: M*{x | Ax <= b} = {x | A*M^-1 x <= b}, with M^-1 = M^T in this case
        A = np.matmul(A, M_proj.T)
        # incorporate effect of shifted center into constraint offset
        b += np.matmul(A, c)

    return hpolyhedron.HPolyhedron(A = A, b = b, validate = False)


@Convert.register_strategy(SetPair('VPolytope', 'Interval'))
def _convert_vpolytope_interval(VP, mode) -> interval.Interval:
    if mode == 'inner':
        if not Represents(VP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_vpolytope_interval(VP, mode = 'outer')
    
    if mode == 'exact':
        if not Represents(VP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        return _convert_vpolytope_interval(VP, mode = 'outer')

    if mode == 'outer':
        return interval.Interval(lb = np.min(VP.V, axis = 0), ub = np.max(VP.V, axis = 0), validate = False)


@Convert.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _convert_vpolytope_zonotope(VP, mode) -> zonotope.Zonotope:
    if VP.number_vertices() == 1:
        return zonotope.Zonotope(c = VP.V[0].flatten(), validate = False)

    if mode == 'inner':
        if not Represents(VP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_vpolytope_zonotope(VP, mode = 'outer')
        
    if mode == 'exact':
        if not Represents(VP, 'Zonotope', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        # don't know how to do exact conversion (method below is exact for 1D and intervals, though)
        if VP.dimension != 1 and not Represents(VP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_vpolytope_zonotope(VP, mode = 'outer')

    if mode == 'outer':
        # convert to interval, then to zonotope
        I = _convert_vpolytope_interval(VP, mode = 'outer')
        return _convert_interval_zonotope(I, mode = 'exact')


@Convert.register_strategy(SetPair('VPolytope', 'HPolyhedron'))
def _convert_vpolytope_hpolyhedron(VP, mode) -> hpolyhedron.HPolyhedron:
    # ***same method for all three modes (exact)
    n = VP.dimension
    if VP.number_vertices() == 1:
        return _convert_point_hpolyhedron(np.reshape(VP.V, (n, )), mode = 'exact')
    
    # shift vertices by mean
    center = np.mean(VP.V, axis = 0)
    V = VP.V - center

    # if polytope is degenerate, we need projection to affine hull and back
    U_, S_, V_ = svd(V.T)
    subspace_dimension = n - np.sum(np.isclose(S_, 0.))
    if subspace_dimension < n:
        # project vertices onto basis and filter out the subspace
        V = np.matmul(U_.T, V.T)
        V = V[0:subspace_dimension].T

    A, b = compute_polytope_halfspaces(V)

    if subspace_dimension < n:
        # project back to original dimension
        h = A.shape[0]
        A = np.vstack((np.hstack((A, np.zeros((h, (n-subspace_dimension))))),
                       np.hstack((np.zeros((2*(n-subspace_dimension), subspace_dimension)),
                                  np.vstack((np.eye(n-subspace_dimension), -np.eye(n-subspace_dimension)))))))
        b = np.hstack((b, np.zeros(2*(n - subspace_dimension))))
        A = np.matmul(A, U_.T)
    b += np.matmul(A, center)
    
    return hpolyhedron.HPolyhedron(A = A, b = b, validate = False)


@Convert.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _convert_hpolyhedron_interval(HP, mode) -> interval.Interval:
    if mode == 'inner':
        if not Represents(HP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_hpolyhedron_interval(HP, mode = 'outer')

    if mode == 'exact':
        if not Represents(HP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        return _convert_hpolyhedron_interval(HP, mode = 'outer')
    
    if mode == 'outer':
        # idea: loop over all 2n -+ basis vectors and use support function value
        n = HP.dimension
        lower_bound = np.zeros(n)
        upper_bound = np.zeros(n)

        basis_vector = np.zeros(n)
        for i in range(n):
            for s in [-1., 1.]:
                basis_vector[i] = s
                value = HP.support_function(basis_vector)[0]
                basis_vector[i] = 0.

                if value == np.inf:
                    raise UnboundedSetError
                elif value == -np.inf:
                    raise EmptySetError
                elif s == -1.:
                    lower_bound[i] = -value
                else:  # s == 1.
                    upper_bound[i] = value
    
        return interval.Interval(lb = lower_bound, ub = upper_bound, validate = False)


@Convert.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _convert_hpolyhedron_zonotope(HP, mode) -> zonotope.Zonotope:
    if mode == 'inner':
        if not Represents(HP, 'Zonotope', rtol = 1e-12, atol = 1e-12)():
            raise NotImplementedError
        return _convert_hpolyhedron_zonotope(HP, mode = 'outer')

    if mode == 'exact':
        if not Represents(HP, 'Zonotope', rtol = 1e-12, atol = 1e-12)():
            raise ExactEvaluationImpossibleError
        elif HP.dimension != 1 and not Represents(HP, 'Interval', rtol = 1e-12, atol = 1e-12)():
            # even if there were an exact method, I do not know about it (method below exact for 1D and intervals)
            raise NotImplementedError
        return _convert_hpolyhedron_zonotope(HP, mode = 'outer')

    if mode == 'outer':
        # convert to interval, then to zonotope
        I = _convert_hpolyhedron_interval(HP, mode = 'outer')
        return _convert_interval_zonotope(I, mode = 'exact')
