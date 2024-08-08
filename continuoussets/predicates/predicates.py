from typing import Union

import numpy as np

# import convex set interface for type checking/docstrings (only as module!)
import continuoussets.convexsets.interface_convexset as cvxset

# default tolerances
import continuoussets.utils.tolerances as tol

# import actual implementation
import continuoussets.predicates.contains as p_contains
import continuoussets.predicates.intersects as p_intersects
import continuoussets.predicates.equals as p_equals

# todo: docstring raises?

# !!! the functions below are exposed to the user


# PREDICATES
def contains(S1: Union[cvxset.IConvexSet, np.ndarray], S2: Union[cvxset.IConvexSet, np.ndarray], *,
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
    return p_contains.Contains(S1, S2, rtol = rtol, atol = atol)()


def equals(S1: Union[cvxset.IConvexSet, np.ndarray], S2: Union[cvxset.IConvexSet, np.ndarray], *,
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
    return p_equals.Equals(S1, S2, rtol = rtol, atol = atol)()


def intersects(S1: Union[cvxset.IConvexSet, np.ndarray], S2: Union[cvxset.IConvexSet, np.ndarray], *,
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
    return p_intersects.Intersects(S1, S2, rtol = rtol, atol = atol)()
