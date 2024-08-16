from __future__ import annotations

from typing import Union, Dict, Tuple, Callable, TYPE_CHECKING
import numpy as np

from continuoussets.unary_operations.interface_unary_operation import IUnaryOperation
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

import continuoussets.convexsets.vpolytope as vpolytope
import continuoussets.convexsets.hpolyhedron as hpolyhedron

from continuoussets.utils.auxiliary import SetPair, remove_duplicate_points, sort_rows
from continuoussets.utils.exceptions import EmptySetError, UnboundedSetError

if __name__ == '__main__':
    print('This is the Represents class.')


class Represents(IUnaryOperation):

    strategies: Dict[Tuple[str, str], Callable] = dict()
    # ordering is not relevant
    ordered_operation = False
    
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S: Union['IConvexSet', np.ndarray], set_class: str, *, rtol: float, atol: float):

        # call superclass constructor
        super().__init__(S, set_class, rtol = rtol, atol = atol)

        # get concrete implementation function
        strategy_key: SetPair = self.get_strategy_key()
        # read out function
        self.func: Callable = Represents.strategies.get(strategy_key)

        # check if given combination is implemented
        if self.func is None:
            raise NotImplementedError

    # evaluate the representation check
    def __call__(self) -> bool:
        return self.func(self.first_operand, **self.kwargs)


@Represents.register_strategy((SetPair('ndarray', 'ndarray'),
                               SetPair('ndarray', 'Interval'),
                               SetPair('ndarray', 'Zonotope'),
                               SetPair('ndarray', 'VPolytope'),
                               SetPair('ndarray', 'HPolyhedron')))
def _represents_point_other(s, rtol, atol) -> bool:
    return True


@Represents.register_strategy(SetPair('Interval', 'ndarray'))
def _represents_interval_point(I, rtol, atol) -> bool:
    return np.allclose(I.diameter(), 0., rtol = rtol, atol = atol)


@Represents.register_strategy((SetPair('Interval', 'Interval'),
                               SetPair('Interval', 'Zonotope'),
                               SetPair('Interval', 'VPolytope'),
                               SetPair('Interval', 'HPolyhedron')))
def _represents_interval_other(I, rtol, atol) -> bool:
    return True


@Represents.register_strategy(SetPair('Zonotope', 'ndarray'))
def _represents_zonotope_point(Z, rtol, atol) -> bool:
    if Z.number_generators() == 0:
        return True
    # estimate size of zonotope
    d = np.sum(np.abs(Z.G), axis = 0)
    return np.allclose(d, 0., rtol = rtol, atol = atol)


@Represents.register_strategy(SetPair('Zonotope', 'Interval'))
def _represents_zonotope_interval(Z, rtol, atol) -> bool:
    if Z.number_generators() == 0:
        return True
    G_abs = np.abs(Z.G)
    return np.allclose(np.sum(G_abs, axis=1), np.max(G_abs, axis=1), rtol = rtol, atol = atol)


@Represents.register_strategy((SetPair('Zonotope', 'Zonotope'),
                               SetPair('Zonotope', 'VPolytope'),
                               SetPair('Zonotope', 'HPolyhedron')))
def _represents_zonotope_other(I, rtol, atol) -> bool:
    return True


@Represents.register_strategy(SetPair('VPolytope', 'ndarray'))
def _represents_vpolytope_point(VP, rtol, atol) -> bool:
    return np.allclose(VP.V - VP.V[0], 0., rtol = rtol, atol = atol)


@Represents.register_strategy(SetPair('VPolytope', 'Interval'))
def _represents_vpolytope_interval(VP, rtol, atol) -> bool:
    # remove all vertices up to the given tolerance from the list of vertices
    VP = VP.compact()
    V_ = vpolytope.VPolytope(V = remove_duplicate_points(VP.V, rtol = rtol, atol = atol), validate = False)

    # intervals must have an number of vertices that is a power of 2
    m = V_.number_vertices()
    if m == 1:
        return True
    elif (m & (m-1)) != 0:
        return False
    max_counter = int(np.log2(m))
    
    # sort vertices
    V = sort_rows(V_.V)

    # there must not be more than two different values for each dimension and equally many of them
    # pattern for ordered vertices of an interval looks as follows (non-degenerate dimensions):
    # 0: [lb, ..., lb, ub, ..., ub]
    # 1: [lb, ..., lb, ub, ..., ub, lb, ..., lb, ub, ..., ub]
    # 2: [lb, ..., lb, ub, ..., ub, lb, ..., lb, ub, ..., ub, ...(repeat)]
    counter = 1
    for i in VP.dimension:
        # check if dimension is degenerate
        if np.allclose(V[:, i], V[0, i], rtol = rtol, atol = atol):
            # dimension is degenerate, do not increment counter
            continue
        
        # dimension is non-degenerate, must follow pattern
        pattern = np.tile(np.repeat([True, False], max_counter-counter+1), counter)

        if not np.allclose(V[pattern, i], V[0, i], rtol = rtol, atol = atol):
            return False
        elif not np.allclose(V[np.invert(pattern), i], V[-1, i], rtol = rtol, atol = atol):
            return False
        
        # increment counter for new pattern
        counter += 1
        
    # all checks ok
    return True


@Represents.register_strategy(SetPair('VPolytope', 'Zonotope'))
def _represents_vpolytope_zonotope(VP, rtol, atol) -> bool:
    # remove all vertices up to the given tolerance from the list of vertices
    VP = VP.compact()
    V_ = vpolytope.VPolytope(V = remove_duplicate_points(VP.V, rtol = rtol, atol = atol), validate = False)

    # due to symmetry, zonotopes always have an even number of vertices (or 1)
    m = V_.number_vertices()
    if m == 1:
        return True
    elif m % 2 != 0:
        return False

    # idea: for each vertex, there must be another vertex across the center
    # 1. subtract the center from the list of vertices
    V = V_.V - V_.center()
    # 2. sort the list of vertices row-wise
    V = sort_rows(V)
    # 3. compare top to bottom (must add up to 0)
    return np.allclose(V[0:int(m/2)] + V[-1:int(m/2)-1:-1], 0., rtol = rtol, atol = atol)


@Represents.register_strategy((SetPair('VPolytope', 'VPolytope'),
                               SetPair('VPolytope', 'HPolyhedron')))
def _represents_vpolytope_other(I, rtol, atol) -> bool:
    return True


@Represents.register_strategy(SetPair('HPolyhedron', 'ndarray'))
def _represents_hpolyhedron_point(HP, rtol, atol) -> bool:
    try:
        c = HP.center()
    except (UnboundedSetError, EmptySetError):
        return False
    # center must fulfill all inequalities with equality and polytope must be bounded
    return np.allclose(np.matmul(HP.A, c), HP.b, rtol = rtol, atol = atol) and HP.bounded()


@Represents.register_strategy(SetPair('HPolyhedron', 'Interval'))
def _represents_hpolyhedron_interval(HP, rtol, atol) -> bool:
    # todo: empty is False (cannot be represented by our Interval class)
    HP = HP.compact(rtol = rtol)
    n, h = HP.dimension, HP.number_constraints()
    if h < 2*n:
        return False
    
    # keep indices for redundancy and for which dimensions are bounded
    index_keep_for_i = np.full((h,), True)
    bounded_dimensions_plus = np.full((n,), False)
    bounded_dimensions_minus = np.full((n,), False)

    # loop over constraints, if not axis-aligned -> must be redundant
    for i in range(h):
        # axis-aligned constraint may only have a single non-zero entry
        non_zero_entry = np.invert(np.isclose(HP.A[i], 0., rtol = rtol, atol = atol))

        if np.sum(non_zero_entry) > 1:
            # check if constraint is redundant
            index_keep_for_i[i] = False
            polyhedron_i = hpolyhedron.HPolyhedron(A = HP.A[index_keep_for_i],
                                                   b = HP.b[index_keep_for_i])
            value = polyhedron_i.support_function(HP.A[i])[0]
            if value > HP.b[i] + atol:
                return False
            # note: we do not have to reset index_keep_for_i, as thee ith constraint is redundant

        else:
            # append this dimension to the respective list of bounded dimensions
            if HP.A[i][non_zero_entry] > 0:
                bounded_dimensions_plus = np.logical_or(bounded_dimensions_plus, non_zero_entry)
            else:
                bounded_dimensions_minus = np.logical_or(bounded_dimensions_minus, non_zero_entry)

    # currently, all dimensions must be bounded
    return np.all(bounded_dimensions_plus) and np.all(bounded_dimensions_minus)


@Represents.register_strategy(SetPair('HPolyhedron', 'Zonotope'))
def _represents_hpolyhedron_zonotope(HP, rtol, atol) -> bool:
    # current idea:
    # - must be bounded (for reasons below, this means at least 2n constraints)
    # - minimal representation must have an even number of constraints
    # - for each constraint, there must be another with factor -1

    HP = HP.compact(rtol = rtol)
    n, h = HP.dimension, HP.number_constraints()
    if h < 2*n or h % 2 != 0 or not HP.bounded():
        return False

    # normalize the constraints
    A_sorted = HP.A / np.reshape(np.linalg.norm(HP.A, ord = 2, axis = 1), (h, 1))
    # sort normalized constraints
    A_sorted = np.sort(A_sorted, axis = 0)

    return np.allclose(A_sorted[0:int(h/2)] + A_sorted[-1:int(h/2)-1:-1], 0., rtol = rtol, atol = atol)
    

@Represents.register_strategy(SetPair('HPolyhedron', 'VPolytope'))
def _represents_hpolyhedron_vpolytope(HP, rtol, atol) -> bool:
    return HP.bounded()
