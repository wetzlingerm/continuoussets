from __future__ import annotations

import inspect
from itertools import product
from typing import Union

import numpy as np

from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils.exceptions import EmptySetError, OutOfBoundsError

if __name__ == '__main__':
    print('This is the Interval class.')


class Interval(ConvexSet):

    def __init__(self, *, lb: Union[np.ndarray, list, float, int] = None,
                 ub: Union[np.ndarray, list, float, int] = None, validate: bool = True) -> Interval:
        """Instantiates an Interval object I = {x | lb <= x <= ub}.

        Args:
            lb (Union[list, np.ndarray], optional): Lower bound. Either lower or upper bound needs to be provided.
            Defaults to None.
            ub (Union[list, np.ndarray], optional): Upper bound. Either lower or upper bound needs to be provided.
            Defaults to None.
            validate (bool, optional): Input argument check. Defaults to True.

        Returns:
            Interval: Bounded, closed, continuous interval.
        """
        if lb is None:
            # use upper bound as lower bound
            lb = ub
        elif ub is None:
            # use lower bound as upper bound
            ub = lb

        # convert to numpy with list of floats (if possible)
        if not isinstance(lb, np.ndarray):
            if isinstance(lb, int) or isinstance(lb, float):
                lb = np.array([float(lb)])
            elif isinstance(lb, list):
                if not all(isinstance(element, float) for element in lb):
                    lb = [float(element) for element in lb]
                lb = np.array(lb)
        if not isinstance(ub, np.ndarray):
            if isinstance(ub, int) or isinstance(ub, float):
                ub = np.array([float(ub)])
            elif isinstance(ub, list):
                if not all(isinstance(element, float) for element in ub):
                    ub = [float(element) for element in ub]
                ub = np.array(ub)

        if self.validate and validate:
            if lb is None and ub is None:
                # at least one bound has to be given
                raise TypeError('Interval:__init__',
                                'No input arguments provided to constructor.')

            # validate input arguments
            if lb.ndim != 1:
                raise ValueError('Interval:__init__',
                                 'Lower bound needs to be a 1D array.')
            elif ub.ndim != 1:
                raise ValueError('Interval:__init__'
                                 'Upper bound needs to be a 1D array.')

            if lb.size != ub.size:
                raise ValueError('Interval:__init__',
                                 'Lower bound and upper bound must have the same size.')

            if not np.all(lb <= ub):
                raise ValueError('Interval:__init__',
                                 'Lower bound needs to be elementwise smaller or equal to upper bound.')

        self.dimension = lb.size
        self.lb = lb.copy()
        self.ub = ub.copy()

    # indexing
    def __getitem__(self, key: Union[int, slice]):
        """Read out the i-th dimension(s) of an Interval I.

        Args:
            key (Union[int, slice]): Dimensions to read out.

        Returns:
            Interval: Composed of the selected dimensions.
        """
        # exception handling done in numpy
        return Interval(lb = self.lb[key], ub = self.ub[key], validate=False)

    # set index
    def __setitem__(self, index: Union[slice, int], value: Interval):
        """Set the i-th dimension of an Interval I.

        Args:
            index (Union[slice, int]): Dimensions to set.
            value (Interval): Lower and upper bounds for the selected dimensions.

        Returns:
            Interval: Selected dimensions are set to new values.
        """
        # exception handling done in numpy
        if isinstance(index, slice):
            self.lb[index] = value.lb
            self.ub[index] = value.ub
        elif isinstance(index, int):
            self.lb[index] = value.lb[0]
            self.ub[index] = value.ub[0]

        return self

    # display
    def __repr__(self) -> str:
        """Representation on the command window.

        Returns:
            str: Description of the Interval object.
        """
        newline = '\n'
        return f'dimension: {self.dimension}{newline}lower bound: {self.lb}{newline}upper bound: {self.ub}'

    # ----------
    # DUNDER METHODS FOR INTERVAL ARITHMETIC
    # ----------

    def __array_ufunc__(self, ufunc, method: str, *args, **kwargs) -> Interval:
        """To enable the correct handling of operations with a numpy object on the left side
        with an Interval object I, i.e., np.array([1., 2.]) * I.

        Args:
            ufunc (np.ufunc): Called ufunc object.
            method (str): Indication which Ufunc method was called. Here: '__call__'.

        Raises:
            NotImplementedError: Only right-operations 'add', 'subtract', 'multiply', and 'divide' are supported.

        Returns:
            Interval: Result of the respective arithmetic operation.
        """
        if ufunc.__name__ == "multiply":
            # should be __rmul__ -> re-order and call __mul__
            return args[1] * args[0]
        elif ufunc.__name__ == "divide":
            # should be __rtruediv__ -> init interval and call __truediv__
            return Interval(lb = args[0], ub = args[0], validate=False) / args[1]
        elif ufunc.__name__ == 'add':
            # should be __radd__ -> re-order and call __add__
            return args[1] + args[0]
        elif ufunc.__name__ == 'subtract':
            # should be __rsub__ -> convert to interval and call __sub__
            return Interval(lb = args[0], ub = args[0]) - args[1]
        else:
            raise NotImplementedError

    # Minkowski sum
    def __add__(self, other: Union[Interval, np.ndarray, list, int, float]) -> Interval:
        """Minkowski sum of an Interval I with another Interval or np.ndarray, list (vector) or int, float (scalar) S.

        Args:
            other (Union[Interval, np.ndarray, list, int, float]):

        Returns:
            Interval: Result of the Minkowski sum.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, Interval):
            lower = self.lb + other.lb
            upper = self.ub + other.ub
        else:
            lower = self.lb + other
            upper = self.ub + other

        return Interval(lb = lower, ub = upper, validate=False)

    # Minkowski sum
    def __radd__(self, other: Union[np.ndarray, list, int, float]) -> Interval:
        """Minkowski sum of a vector or scalar with an Interval.

        Args:
            other (Union[np.ndarray, list, int, float]): Vector or scalar.

        Returns:
            Interval: Result of the Minkowski sum.
        """
        return self + other

    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        """Set equality of an Interval I with another set or vector S.
        Defined as forall i in I: i in S and forall s in S: s in I?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.

        Returns:
            bool: Set equality.
        """
        self._checkOtherOperand(other, check_dimension = False)

        if ((isinstance(other, ConvexSet) and self.dimension != other.dimension)
                or (isinstance(other, np.ndarray) and self.dimension != other.shape[0])):
            return False
        elif isinstance(other, np.ndarray):
            return self == Interval(lb = other, ub = other)
        elif isinstance(other, Interval):
            return np.allclose(self.lb, other.lb) and np.allclose(self.ub, other.ub)
        elif isinstance(other, ConvexSet):
            return other.represents('Interval') and self == Interval(**other.interval(), validate=False)

    # element-wise multiplication
    def __mul__(self, other: Union[Interval, np.ndarray, list, int, float]) -> Interval:
        """Elementwise multiplication of an Interval I with another Interval, vector or scalar S.

        Args:
            other (Union[Interval, np.ndarray, list, int, float]): Right factor Interval, vector or scalar.

        Returns:
            Interval: Result of the multiplication.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, Interval):
            possible_values = np.vstack((self.lb * other.lb,
                                         self.lb * other.ub,
                                         self.ub * other.lb,
                                         self.ub * other.ub))
        else:
            possible_values = np.vstack((other * self.lb,
                                         other * self.ub))

        lower = np.min(possible_values, axis=0)
        upper = np.max(possible_values, axis=0)
        return Interval(lb = lower, ub = upper, validate=False)

    # element-wise multiplication
    def __rmul__(self, other: Union[np.ndarray, list, int, float]) -> Interval:
        """Elementwise multiplication of a vector or scalar S with an Interval I.

        Args:
            other (Union[np.ndarray, list, int, float]): Left factor vector or scalar.

        Returns:
            Interval: Result of the multiplication.
        """
        return self * other

    # unary minus
    def __neg__(self) -> Interval:
        """Unary minus operator.

        Returns:
            Interval: Input Interval times -1.
        """
        return Interval(lb = -self.ub, ub = -self.lb, validate=False)

    # unary plus
    def __pos__(self) -> Interval:
        """Unary plus operator.

        Returns:
            Interval: Same as input Interval.
        """
        return Interval(lb = self.lb, ub = self.ub, validate=False)

    # exponentiation
    def __pow__(self, power: Union[np.ndarray, list, float, int]) -> Interval:
        """Elementwise exponentiation of an Interval I by a vector or scalar S.

        Args:
            power (Union[np.ndarray, list, float, int]): Exponent vector or scalar.

        Raises:
            ValueError: Dimensions containing zero cannot be exponentiated by negative exponents.
            ValueError: Negative bases cannot be exponentiated by non-integer exponents.

        Returns:
            Interval: Result of the exponentiation.
        """
        if isinstance(power, int) or isinstance(power, float):
            power = np.repeat(np.array(power), self.dimension)
        elif isinstance(power, list):
            power = np.array(power)

        if np.all(power == 0):
            return Interval(lb = np.ones(self.dimension), ub = np.ones(self.dimension), validate=False)
        elif np.all(power == 1):
            return Interval(lb = self.lb, ub = self.ub, validate=False)

        # init bounds
        lower = np.zeros(self.dimension)
        upper = np.zeros(self.dimension)

        # base contains zero and exponent is negative -> Inf
        if np.any(np.all(np.vstack((self.lb <= np.zeros(self.dimension), self.ub >= np.zeros(self.dimension), power < 0)),
                         axis=0)):
            raise ValueError('Interval:__pow__',
                             'Exponentiation of 0 with a negative exponent.')

        # negative exponents: base cannot be negative (see above)
        index_neg = power < 0
        lower[index_neg] = self.ub[index_neg] ** power[index_neg]
        upper[index_neg] = self.lb[index_neg] ** power[index_neg]
        remaining_indices = np.logical_not(index_neg)

        # non-integer exponents
        index_noninteger = np.logical_and(remaining_indices, np.mod(power, 1) != 0)
        if np.any(np.all(np.vstack((index_noninteger, self.lb < 0)), axis=0)):
            raise ValueError('Interval:__pow__',
                             'Exponentiation with non-integer number and negative base.')
        lower[index_noninteger] = self.lb[index_noninteger] ** power[index_noninteger]
        upper[index_noninteger] = self.ub[index_noninteger] ** power[index_noninteger]
        remaining_indices = np.logical_and(remaining_indices, np.logical_not(index_noninteger))

        # even exponent -> resulting interval is non-negative
        index_even = np.logical_and(remaining_indices, np.mod(power, 2) == 0)
        lower[index_even] = np.minimum(np.abs(self.lb), np.abs(self.ub))[index_even] ** power[index_even]
        upper[index_even] = np.maximum(np.abs(self.lb), np.abs(self.ub))[index_even] ** power[index_even]
        remaining_indices = np.logical_and(remaining_indices, np.logical_not(index_even))

        # odd exponents
        index_odd = remaining_indices
        lower[index_odd] = self.lb[index_odd] ** power[index_odd]
        upper[index_odd] = self.ub[index_odd] ** power[index_odd]

        return Interval(lb = lower, ub = upper, validate=False)

    # interval arithmetic subtraction (not Minkowski difference!)
    def __sub__(self, other: Union[Interval, np.ndarray, list, int, float]) -> Interval:
        """Subtraction of an Interval, vector or scalar S from an Interval I.
        Defined as Minkowski sum of I and (-S).
        Note: This is not the Minkowski difference!

        Args:
            other (Union[Interval, np.ndarray, list, int, float]): Subtrahend set, vector or scalar.

        Returns:
            Interval: Result of the Minkowski sum.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, Interval):
            lower = self.lb - other.ub
            upper = self.ub - other.lb
        else:
            lower = self.lb - other
            upper = self.ub - other

        return Interval(lb = lower, ub = upper, validate=False)

    # interval arithmetic subtraction (not Minkowski difference!)
    def __rsub__(self, other: Union[np.ndarray, list, int, float]) -> Interval:
        """Subtraction of an Interval I from a vector or scalar S.
        Defined as Minkowski sum of I and (-S).
        Note: This is not the Minkowski difference!

        Args:
            other (Union[np.ndarray, list, int, float]): Minuend vector or scalar.

        Returns:
            Interval: Result of the Minkowski sum.
        """
        if isinstance(other, int) or isinstance(other, float):
            other = np.repeat(other, self.dimension)
        elif isinstance(other, list):
            other = np.array(other)
        minuend = Interval(lb = other, ub = other, validate=False)
        return minuend - self

    # division
    def __truediv__(self, other: Union[Interval, np.ndarray, list, int, float]) -> Interval:
        """Elementwise division of an Interval I by an Interval, vector, or scalar S.

        Args:
            other (Union[Interval, np.ndarray, list, int, float]): Denominator Interval, vector, or scalar.

        Raises:
            ZeroDivisionError: Interval denominator contains 0 in at least one dimension.
            ZeroDivisionError: Vector denominator is 0 in at least one dimension. Scalar denominator is zero.

        Returns:
            Interval: Result of the division.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, Interval):
            if np.any(other.contains(np.zeros(other.dimension))):
                raise ZeroDivisionError

            stacked_result = np.vstack((self.lb / other.lb,
                                        self.lb / other.ub,
                                        self.ub / other.lb,
                                        self.ub / other.ub))
            lower = np.min(stacked_result, axis=0)
            upper = np.max(stacked_result, axis=0)
        else:
            if np.any(np.isclose(other, 0)):
                raise ZeroDivisionError

            if isinstance(other, int) or isinstance(other, float):
                if other < 0:
                    lower = self.ub / other
                    upper = self.lb / other
                else:
                    lower = self.lb / other
                    upper = self.ub / other
            else:  # other is list or np.ndarray
                stacked_result = np.vstack((self.lb / other, self.ub / other))
                lower = np.min(stacked_result, axis=0)
                upper = np.max(stacked_result, axis=0)

        return Interval(lb = lower, ub = upper, validate=False)

    # division
    def __rtruediv__(self, other: Union[np.ndarray, list, int, float]) -> Interval:
        """Elementwise division of a vector or scalar S by an Interval I.

        Args:
            other (Union[Interval, np.ndarray, list, int, float]): Numerator vector, or scalar.

        Returns:
            Interval: Result of the division.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, int) or isinstance(other, float):
            other = np.repeat(other, self.dimension)
        elif isinstance(other, list):
            other = np.array(other)
        dividend = Interval(lb = other, ub = other, validate=False)
        return dividend / self

    # INTERVAL ARITHMETIC OPERATIONS
    # absolute value
    def absolute_value(self) -> Interval:
        """Absolute value of an Interval I.
        Defined as |I| = {x | min{|lb|, |ub|} <= x <= max{|lb|, |ub|}}

        Returns:
            Interval: Absolute value of the Interval.
        """
        stacked_abs = np.vstack((np.abs(self.lb), np.abs(self.ub)))
        return Interval(lb = np.min(stacked_abs, axis=0),
                        ub = np.max(stacked_abs, axis=0), validate=False)

    # arccosine
    def arccos(self) -> Interval:
        """Elementwise evaluation of the arccosine of an Interval I.

        Raises:
            OutOfBoundsError: Interval must be bounded between -1 and 1.

        Returns:
            Interval: Range of the arccosine over the Interval.
        """
        outside_bounds = np.vstack((self.lb < -1, self.ub > 1))
        if np.any(outside_bounds):
            dim = np.where(outside_bounds)[1][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = f"{(-1, 1)}")

        return Interval(lb = np.arccos(self.ub), ub = np.arccos(self.lb), validate=False)

    # arcsine
    def arcsin(self) -> Interval:
        """Elementwise evaluation of the arcsine of an Interval I.

        Raises:
            OutOfBoundsError: Interval must be bounded between -1 and 1.

        Returns:
            Interval: Range of the arcsine over the Interval.
        """
        outside_bounds = np.vstack((self.lb < -1, self.ub > 1))
        if np.any(outside_bounds):
            dim = np.where(outside_bounds)[1][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = f"{(-1, 1)}")

        return Interval(lb = np.arcsin(self.lb), ub = np.arcsin(self.ub), validate=False)

    # arctangent
    def arctan(self) -> Interval:
        """Elementwise evaluation of the arctangent of an Interval I.

        Returns:
            Interval: Range of the arctangent over the Interval.
        """
        return Interval(lb = np.arctan(self.lb), ub = np.arctan(self.ub), validate=False)

    # cosine
    def cos(self) -> Interval:
        """Elementwise evaluation of the cosine of an Interval I.

        Returns:
            Interval: Range of the cosine over the Interval.
        """

        lower = -((self - np.pi)._maxcos())
        upper = self._maxcos()
        return Interval(lb = lower, ub = upper, validate=False)

    def _maxcos(self) -> np.ndarray:
        """Adrian's magic helper function for the evaluation of the cosine over an Interval.

        Returns:
            np.ndarray: Auxiliary value (see Interval.cos).
        """
        # periods
        k = np.ceil(self.lb/(2.*np.pi))
        # update lower and upper bound
        lower = self.lb - 2.*np.pi * k
        upper = self.ub - 2.*np.pi * k
        # compute max
        return np.maximum(np.maximum(np.cos(lower), np.cos(upper)), np.sign(upper))

    # diameter
    def diameter(self) -> np.ndarray:
        """Computation of the diameter of an Interval I.
        Defined as (ub - lb).

        Returns:
            np.ndarray: Diameter.
        """
        return self.ub - self.lb

    # dot product
    def dot(self, other: Union[Interval, np.ndarray]) -> Interval:
        """Computation of the dot of an Interval I with another Interval or vector S.
        Defined as sum_i I_i^T * S_i.

        Args:
            other (Union[Interval, np.ndarray]): Interval or vector.

        Returns:
            Interval: Range of the dot product.
        """
        self._checkIntervalArithmetic(other)

        if isinstance(other, np.ndarray):
            possible_values = np.vstack((other * self.lb,
                                         other * self.ub))
        elif isinstance(other, Interval):
            possible_values = np.vstack((self.lb * other.lb,
                                         self.lb * other.ub,
                                         self.ub * other.lb,
                                         self.ub * other.ub))

        lower = np.sum(np.min(possible_values, axis=0))
        upper = np.sum(np.max(possible_values, axis=0))
        return Interval(lb = np.array([lower]), ub = np.array([upper]), validate=False)

    # natural logarithm
    def log(self) -> Interval:
        """Elementwise evaluation of the natural logarithm of an Interval I.

        Returns:
            Interval: Range of the natural logarithm over the Interval.
        """
        outside_bounds = (self.lb <= 0)
        if np.any(outside_bounds):
            dim = np.where(outside_bounds)[0][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = f"{(np.finfo(np.float64).eps, np.inf)}")

        return Interval(lb = np.log(self.lb), ub = np.log(self.ub), validate=False)

    # logarithm with base 10
    def log10(self) -> Interval:
        """Elementwise evaluation of the logarithm with base 10 of an Interval I.

        Returns:
            Interval: Range of the logartihm with base 10 over the Interval.
        """
        outside_bounds = (self.lb <= 0)
        if np.any(outside_bounds):
            dim = np.where(outside_bounds)[0][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = f"{(np.finfo(np.float64).eps, np.inf)}")

        return Interval(lb = np.log10(self.lb), ub = np.log10(self.ub), validate=False)

    # sine
    def sin(self) -> Interval:
        """Elementwise evaluation of the sine of an Interval I.

        Returns:
            Interval: Range of the sine over the Interval.
        """
        return (self - 0.5 * np.pi).cos()

    # square root
    def sqrt(self) -> Interval:
        """Elementwise evaluation of the square root of an Interval I.

        Returns:
            Interval: Range of the square root over the Interval.
        """
        outside_bounds = (self.lb < 0)
        if np.any(outside_bounds):
            dim = np.where(outside_bounds)[0][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = f"{(0, np.inf)}")

        return Interval(lb = np.sqrt(self.lb), ub = np.sqrt(self.ub), validate=False)

    # tangent
    def tan(self) -> Interval:
        """Elementwise evaluation of the tangent of an Interval I.

        Returns:
            Interval: Range of the tangent over the Interval.
        """
        # ensure that diameter is not larger than pi
        d = self.diameter() >= np.pi
        if np.any(d):
            dim = np.where(d)[0][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = "Any whole-number multiple of (0, 2*pi)")

        # compute tangent elementwise
        lower_tan = np.tan(self.lb)
        upper_tan = np.tan(self.ub)

        # check if there is a jump
        jump = upper_tan < lower_tan
        if np.any(jump):
            dim = np.where(jump)[0][0]
            raise OutOfBoundsError(dimension = dim,
                                   given_range = (self.lb[dim], self.ub[dim]),
                                   valid_range = "Any whole-number multiple of (0, 2*pi)")

        return Interval(lb = lower_tan, ub = upper_tan, validate=False)

    # ----------
    # SET OPERATIONS
    # ----------

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of an Interval I in a given direction.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Raises:
            NotImplementedError: Interval must contain the origin.
            NotImplementedError: Interval must be non-degenerate

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        # limit to intervals containing the origin for now...
        if not self.contains(np.zeros(self.dimension)):
            raise NotImplementedError
        elif np.any(self.diameter() == 0):
            # exclude degenerate for now...
            raise NotImplementedError

        # normalize direction
        direction = direction / np.linalg.norm(direction)
        # for positive values, extract upper bound; for negative values, extract lower bound
        bound = self.lb + np.maximum(np.sign(direction), 0.) * self.diameter()
        # compute factor of limiting dimension
        ratio = bound[np.logical_not(direction == 0)] / direction[np.logical_not(direction == 0)]
        # take minimum value (excluding -Inf)
        ratio = np.min(ratio[np.logical_not(np.isinf(ratio))])
        # multiply (normalized) direction with that factor
        return direction * ratio

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> Interval:
        """Cartesian product of an Interval I and another set or vector S.
        Defined as {[a^T s^T]^T | a in I, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'outer'.

        Raises:
            NotImplementedError: Inner approximation and exact evaluation not implemented unless other represents an Interval.

        Returns:
            Interval: Result of the Cartesian product.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if mode in ['inner', 'exact'] and not other.represents('Interval'):
            raise NotImplementedError

        if isinstance(other, Interval):
            return Interval(lb = np.hstack((self.lb, other.lb)),
                            ub = np.hstack((self.ub, other.ub)), validate=False)
        elif isinstance(other, ConvexSet):
            return self.cartesian_product(Interval(**other.interval(mode = mode), validate=False))
        else:
            return Interval(lb = np.hstack((self.lb, other)),
                            ub = np.hstack((self.ub, other)), validate=False)

    # center
    def center(self) -> np.ndarray:
        """Center of an Interval I.
        Defined as 0.5*(lb + ub).

        Returns:
            np.ndarray: Center of the Interval.
        """
        return 0.5 * (self.lb + self.ub)

    # compact
    def compact(self, *, rtol: float = 0.0) -> Interval:
        """Minimal representation of an Interval I. (Merely implemented for duck typing purposes.)

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 0.0.

        Returns:
            Interval: Same as input Interval.
        """
        return Interval(lb = self.lb, ub = self.ub)

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        """Checks containment of a set or vector S in an Interval I.
        Defined as forall s in S: s in I?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.

        Returns:
            bool: Containment status.
        """
        self._checkOtherOperand(other)

        if isinstance(other, Interval):
            return np.all(self.lb <= other.lb) and np.all(self.ub >= other.ub)
        elif isinstance(other, ConvexSet):
            return self.contains(Interval(**other.interval(), validate=False))
        else:
            return np.all(self.lb <= other) and np.all(self.ub >= other)

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> Interval:
        """Convex hull of an Interval I and another set or vector S.
        Defined as {lambda*a + (1-lambda)*s | a in I, s in S, lambda in [0,1]}

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of operation: 'inner', 'exact', 'outer'. Defaults to 'outer'.

        Raises:
            NotImplementedError: Inner approximation and exact evaluation not implemented in the general case.

        Returns:
            Interval: Result of the convex hull.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if mode in ['inner', 'exact'] and not (self.contains(other) or other.contains(self)):
            raise NotImplementedError

        if isinstance(other, Interval):
            return Interval(lb = np.minimum(self.lb, other.lb),
                            ub = np.maximum(self.ub, other.ub), validate=False)
        elif isinstance(other, ConvexSet):
            return self.convex_hull(Interval(**other.interval(mode = mode), validate=False))
        else:
            return Interval(lb = np.minimum(self.lb, other),
                            ub = np.maximum(self.ub, other), validate=False)

    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        """Checks if an Interval I intersects another set of vector S.
        Defined as exists s in I: s in S?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.

        Returns:
            bool: Result of the intersection check.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)
        elif isinstance(other, Interval):
            return np.any(np.logical_not(np.any(np.vstack((other.ub <= self.lb, other.lb >= self.ub)), axis=0)))
        elif isinstance(other, ConvexSet):
            return other.intersects(self)

    # conversion to interval
    def interval(self, *, mode: str = 'exact') -> dict:
        """Overloaded conversion to Interval.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an Interval object.
        """
        self._checkMode(mode)

        return {'lb': self.lb, 'ub': self.ub}

    # linear map
    def matmul(self, matrix: np.ndarray) -> Interval:
        """Linear map of an interval I by a matrix (np.ndarray).
        Defined as {M s | s in I}.

        Args:
            matrix (np.ndarray): Matrix for left-multiplication.

        Returns:
            Interval: Result of the matrix multiplication.
        """
        self._checkMatrix(matrix)

        if isinstance(matrix, np.ndarray):
            matrix_lb = matrix * self.lb
            matrix_ub = matrix * self.ub
            lower = np.sum(np.minimum(matrix_lb, matrix_ub), axis=1)
            upper = np.sum(np.maximum(matrix_lb, matrix_ub), axis=1)
        # TODO: matrix has to be IntervalMatrix object!

        return Interval(lb = lower, ub = upper, validate=False)

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> Interval:
        """Minkowski sum between an Interval I and another set of vector S.
        Defined as {a + s | a in I, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Summand.
            mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'outer'.

        Returns:
            Interval: Result of the Minkowski sum.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            lower = self.lb + other
            upper = self.ub + other
        elif isinstance(other, Interval):
            lower = self.lb + other.lb
            upper = self.ub + other.ub
        else:
            return self + Interval(**other.interval(mode = mode), validate=False)

        return Interval(lb = lower, ub = upper, validate=False)

    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> Interval:
        """Minkowski difference between an Interval I and another set or vector S.
        Defined as {s | s + S in I}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            EmptySetError: Result is the empty set.

        Returns:
            Interval: Result of the Minkowski difference.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # special case: Minkowski difference with a vector
        if isinstance(other, np.ndarray):
            return self - other

        # convert subtrahend to interval
        if not isinstance(other, Interval):
            other = Interval(**other.interval(mode = 'outer'))

        # check diameters
        other_diameter = other.diameter()
        if np.any(self.diameter() < other_diameter):
            raise EmptySetError

        other_center = other.center()
        return Interval(lb = self.lb - other_center + 0.5*other_diameter,
                        ub = self.ub - other_center - 0.5*other_diameter, validate=False)

    # projection onto subspace
    def project(self, *, axis: tuple) -> Interval:
        """Projection of an Interval I onto a subspace.

        Args:
            axis (tuple): Subspace for projection.

        Returns:
            Interval: Projected Interval.
        """
        self._checkSubspace(axis)

        # convert tuples to lists for indexing
        return Interval(lb = self.lb[list(axis)], ub = self.ub[list(axis)], validate=False)

    # reduction (implement for overloading)
    def reduce(self) -> Interval:
        """Reduction of the set representation size of an Interval I. (Merely implemented for duck typing purposes.)

        Returns:
            Interval: Interval with reduced set representation size.
        """
        return Interval(lb = self.lb, ub = self.ub, validate=False)

    # representation by other set representation
    def represents(self, set_class: str) -> bool:
        """Check if an interval I can also be equivalently represented using another ConvexSet class.

        Args:
            set_class (str): Name of another ConvexSet class.

        Returns:
            bool: Representation possible.
        """
        self._checkSetClass(set_class)

        return True

    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Support function evaluation of an Interval I in a direction d.
        Support value defined as max_{s in I} d^T * s.
        Support vector defined as arg max_{s in I} d^T * s.

        Args:
            direction (np.ndarray): Direction along which to evaluate the support function.

        Returns:
            tuple[float, np.ndarray]: Support value and support vector.
        """
        self._checkOtherOperand(direction)

        # support vector is vertex
        vector = self.lb.copy()
        index_ub = np.sign(direction) == 1
        vector[index_ub] = self.ub.copy()[index_ub]

        # support function value via dot product
        value = np.dot(direction, vector)
        return (value, vector)

    # symmetric interval (around 0)
    def symm(self) -> Interval:
        """Symmetric interval hull centered at 0 of an Interval I.

        Returns:
            Interval: Enclosing symmetric Interval.
        """
        bound = np.max(np.vstack((np.abs(self.lb), np.abs(self.ub))), axis=0)
        return Interval(lb = -bound, ub = bound, validate=False)

    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of an Interval I.

        Returns:
            np.ndarray: 2D array containing vertices as columns.
        """
        # reformat so that each dimension is a single np.ndarray (required for combinations below)
        bounds_per_dimension = np.vsplit(np.vstack((self.lb, self.ub)).transpose(), self.dimension)

        # remove second dimension for individual dimensions
        var = [x.flatten() for x in bounds_per_dimension]

        # flatten dimensions where lower bound equals the upper bound, write in tuple to unpack for itertools.product call
        t = tuple(x if x[0] != x[1] else np.array([x[0]]) for x in var)

        # enumerate all combinations
        all_combinations = product(*t)

        # stack combinations, transpose so that vertices are columns
        V = np.transpose(np.vstack([np.array(x) for x in all_combinations]))
        return V

    # volume computation
    def volume(self) -> float:
        """Computation of the volume of an n-dimensional Interval I.
        Defined as prod_{i=1}^{n} (ub_i - lb_i).

        Returns:
            float: Volume.
        """
        return np.prod(self.diameter())

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'exact') -> dict:
        """Conversion of an Interval I to a Zonotope Z.

        Args:
            mode (str, optional): Approximation of conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: keyword arguments for instantiation of a Zonotope object
        """
        self._checkMode(mode)

        # for consistency, support modes 'exact', 'outer', 'inner'
        # since every interval is a zonotope, these yield the same results
        generators = np.diag(self.diameter())
        return {'c': self.center(), 'G': 0.5*generators[:, ~np.all(generators == 0, axis=0)]}

    # check function
    def _checkIntervalArithmetic(self, other):
        # wrapper ensures that the other operand is either int, float, list, np.ndarray or Interval object
        if self.validate:
            if (not isinstance(other, int) and not isinstance(other, float) and not isinstance(other, list)
                    and not isinstance(other, np.ndarray) and not isinstance(other, Interval)):
                raise TypeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                f'Other operand must be of type int, float, list, np.ndarray or Interval')
