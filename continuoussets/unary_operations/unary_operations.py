from typing import Union, TYPE_CHECKING

import numpy as np

# import convex set interface for type checking/docstrings
if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet

# default tolerances
import continuoussets.utils.tolerances as tol

# import actual implementation
import continuoussets.unary_operations.represents as op_represents
import continuoussets.unary_operations.convert as op_convert

# todo: docstring raises?

# !!! the functions below are exposed to the user


# PARAMETRIZED UNARY OPERATIONS
def represents(S: Union['IConvexSet', np.ndarray],
               set_class: str, *,
               rtol: float = tol.REPRESENTS_RTOL,
               atol: float = tol.REPRESENTS_ATOL) -> bool:
    """Checks if a IConvexSet S can also be equivalently represented by another IConvexSet class or a point.

    Args:
        S (Union[IConvexSet, np.ndarray]): Set or vector.
        set_class (str): Name of another IConvexSet class or 'Point'.
        rtol (float, optional): Relative tolerance. Defaults to REPRESENTS_RTOL.
        atol (float, optional): Absolute tolerance. Defaults to REPRESENTS_ATOL.

    Returns:
        bool: Representation possible.
    """
    # call implementation
    return op_represents.Represents(S, set_class, rtol = rtol, atol = atol)()


def convert(S: Union['IConvexSet', np.ndarray],
            set_class: str, *,
            mode: str = 'exact') -> bool:
    """Converts a IConvexSet S to another IConvexSet class or a point.

    Args:
        S (Union[IConvexSet, np.ndarray]): Set or vector.
        set_class (str): Name of another IConvexSet class or 'Point'.
        mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        bool: Representation possible.
    """
    # call implementation
    return op_convert.Convert(S, set_class, mode = mode)()
