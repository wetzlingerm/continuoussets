# import convex set interface for type checking/docstrings (only as module!)
import continuoussets.convexsets.interface_convexset as cs

# import actual implementation
import continuoussets.conversions.convert as c_convert

# todo: docstring raises?

# !!! the functions below are exposed to the user


# CONVERSION
def convert(S: cs.IConvexSet, set_class: str, *, mode: str = 'exact') -> cs.IConvexSet:
    """Converts a IConvexSet to another IConvexSet.

    Args:
        S (IConvexSet): Set.
        set_class (str): Name of target set representation.
        mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    Returns:
        IConvexSet: Converted set.
    """
    # call implementation
    return c_convert.Convert(S, set_class = set_class, mode = mode)()
