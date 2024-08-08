from typing import Union

import numpy as np

# import convex set interface for type checking/docstrings (only as module!)
import continuoussets.convexsets.interface_convexset as cs

# import actual implementation
import continuoussets.set_operations.cartesian_product as ops_cartesian_product
import continuoussets.set_operations.convex_hull as ops_convex_hull
import continuoussets.set_operations.minkowski_sum as ops_minkowski_sum
import continuoussets.set_operations.minkowski_difference as ops_minkowski_difference

# todo: docstring raises

# !!! the functions here are exposed to the user


# BINARY OPERATIONS
def cartesian_product(S1: Union[cs.IConvexSet, np.ndarray],
                      S2: Union[cs.IConvexSet, np.ndarray],
                      mode: str = 'exact') -> cs.IConvexSet:
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
    return ops_cartesian_product.CartesianProduct(S1, S2, mode = mode)


def convex_hull(S1: Union[cs.IConvexSet, np.ndarray],
                S2: Union[cs.IConvexSet, np.ndarray],
                mode: str = 'exact') -> cs.IConvexSet:
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
    return ops_convex_hull.ConvexHull(S1, S2, mode = mode)


def minkowski_difference(S1: Union[cs.IConvexSet, np.ndarray],
                         S2: Union[cs.IConvexSet, np.ndarray],
                         mode: str = 'exact') -> cs.IConvexSet:
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
    return ops_minkowski_difference.MinkowskiDifference(S1, S2, mode = mode)


def minkowski_sum(S1: Union[cs.IConvexSet, np.ndarray],
                  S2: Union[cs.IConvexSet, np.ndarray],
                  mode: str = 'exact') -> cs.IConvexSet:
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
    return ops_minkowski_sum.MinkowskiSum(S1, S2, mode = mode)
