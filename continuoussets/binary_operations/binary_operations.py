from typing import Union, TYPE_CHECKING

import numpy as np

# import convex set interface for type checking/docstrings
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

# default tolerances
import continuoussets.utils.tolerances as tol

# import actual implementation
import continuoussets.binary_operations.contains as op_contains
import continuoussets.binary_operations.intersects as op_intersects
import continuoussets.binary_operations.equals as op_equals
import continuoussets.binary_operations.cartesian_product as op_cartesian_product
import continuoussets.binary_operations.convex_hull as op_convex_hull
import continuoussets.binary_operations.minkowski_difference as op_minkowski_difference
import continuoussets.binary_operations.minkowski_sum as op_minkowski_sum


# todo: docstring raises?

# !!! the functions below are exposed to the user


# BINARY OPERATIONS
def cartesian_product(S1: Union['IConvexSet', np.ndarray],
                      S2: Union['IConvexSet', np.ndarray],
                      mode: str = 'exact') -> 'IConvexSet':
    """Cartesian product of two IConvexSet or vectors S1 and S2.
    Defined as: {[s1^T s2^T]^T | s1 in S1, s2 in S2}.

    Args:
        S1 (Union[IConvexSet, np.ndarray]): Set or vector.
        S2 (Union[IConvexSet, np.ndarray]): Set or vector.
        mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        IConvexSet: Cartesian product of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    """
    # call implementation
    return op_cartesian_product.CartesianProduct(S1, S2, mode = mode)


def convex_hull(S1: Union['IConvexSet', np.ndarray],
                S2: Union['IConvexSet', np.ndarray],
                mode: str = 'exact') -> 'IConvexSet':
    """Convex hull of two IConvexSet or vectors S1 and S2.
    Defined as: {lambda*s1 + (1-lambda)*s2 | s1 in S2, s2 in S2, lambda in [0,1]}.

    Args:
        S1 (Union[IConvexSet, np.ndarray]): Set or vector.
        S2 (Union[IConvexSet, np.ndarray]): Set or vector.
        mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        IConvexSet: Convex hull of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    """
    # call implementation
    return op_convex_hull.ConvexHull(S1, S2, mode = mode)


def minkowski_difference(S1: Union['IConvexSet', np.ndarray],
                         S2: Union['IConvexSet', np.ndarray],
                         mode: str = 'exact') -> 'IConvexSet':
    """Minkowski difference of two IConvexSet or vectors S1 and S2.
    Defined as {s | s + S2 in S1}.

    Args:
        S1 (Union[IConvexSet, np.ndarray]): Set or vector.
        S2 (Union[IConvexSet, np.ndarray]): Set or vector.
        mode (str, optional): Approximation of the result:: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        IConvexSet: Minkowski difference of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    """
    # call implementation
    return op_minkowski_difference.MinkowskiDifference(S1, S2, mode = mode)


def minkowski_sum(S1: Union['IConvexSet', np.ndarray],
                  S2: Union['IConvexSet', np.ndarray],
                  mode: str = 'exact') -> 'IConvexSet':
    """Minkowski sum of two IConvexSet or vectors S1 and S2.
    Defined as: {s1 + s2 | s1 in S1, s2 in S2}.

    Args:
        S1 (Union[IConvexSet, np.ndarray]): Set or vector.
        S2 (Union[IConvexSet, np.ndarray]): Set or vector.
        mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        IConvexSet: Minkowski sum of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    """
    # call implementation
    return op_minkowski_sum.MinkowskiSum(S1, S2, mode = mode)


# PREDICATES
def contains(S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
             rtol: float = tol.CONTAINS_RTOL,
             atol: float = tol.CONTAINS_ATOL) -> bool:
    """Checks containment of an IConvexSet or vector S2 in an IConvexSet S1.
    Defined as: forall s2 in S2: s2 in S1?

    Args:
        other (Union[IConvexSet, np.ndarray]): Set or vector.
        rtol (float, optional): Relative tolerance. Defaults to CONTAINS_RTOL.
        atol (float, optional): Absolute tolerance. Defaults to CONTAINS_ATOL.

    Returns:
        bool: Containment.
    """
    # call implementation
    return op_contains.Contains(S1, S2, rtol = rtol, atol = atol)()


def equals(S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
           rtol: float = tol.EQUALS_RTOL,
           atol: float = tol.EQUALS_ATOL) -> bool:
    """Checks set equality of two IConvexSet derived objects S1 and S2.
    Defined as: forall s1 in S1: s1 in S2 and forall s2 in S2: s2 in S1?

    Args:
        other (Union[IConvexSet, np.ndarray]): Set or vector.
        rtol (float, optional): Relative tolerance. Defaults to EQUALS_RTOL.
        atol (float, optional): Absolute tolerance. Defaults to EQUALS_ATOL.

    Returns:
        bool: Set equality.
    """
    # call implementation
    return op_equals.Equals(S1, S2, rtol = rtol, atol = atol)()


def intersects(S1: Union['IConvexSet', np.ndarray], S2: Union['IConvexSet', np.ndarray], *,
               rtol: float = tol.INTERSECTS_RTOL,
               atol: float = tol.INTERSECTS_ATOL) -> bool:
    """Checks whether the intersection of an IConvexSet or vector S1 with an IConvexSet or vector S2 is non-empty.
    Defined as: exists s1 in S1: s1 in S2?

    Args:
        other (Union[IConvexSet, np.ndarray]): Set or vector.
        rtol (float, optional): Relative tolerance. Defaults to INTERSECTS_RTOL.
        atol (float, optional): Absolute tolerance. Defaults to INTERSECTS_ATOL.

    Returns:
        bool: Non-emptiness of intersection.
    """
    # call implementation
    return op_intersects.Intersects(S1, S2, rtol = rtol, atol = atol)()
