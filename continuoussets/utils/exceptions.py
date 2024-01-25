import inspect

if __name__ == '__main__':
    "This is a utilities file for custom Exceptions"


class OutOfBoundsError(Exception):

    def __init__(self, *, dimension: int, given_range: tuple, valid_range: str):
        """Exception raised if range of an Interval is out of bounds for the evaluation of a function.

        Args:
            dimension (int): Dimension which is out of bounds.
            given_range (tuple): Range of the violating dimension.
            valid_range (str): Text describing the valid range.
        """
        newline = "\n"
        message = f"Interval.{inspect.stack()[1].function}:{newline}\
                    Range of Interval in dimension {dimension} is out of bounds.{newline}\
                    Given range: {given_range}, valid range: {valid_range}."
        super().__init__(message)


class EmptySetError(Exception):

    def __init__(self):
        """Exception raised if the result of a set operation is the empty set.
        """
        newline = "\n"
        message = f"{inspect.stack()[1].function}:{newline}\
                    The result of this set operation is the empty set."
        super().__init__(message)


class OtherFunctionError(Exception):

    def __init__(self, types: tuple, other_function: str):
        """Exception raised if another function likely achieves the desired result.

        Args:
            types (tuple): Types that have been used to call the function.
            other_function (str): Function that should be called instead.
        """
        newline = "\n"
        message = f"{inspect.stack()[1].function}:{newline}\
                    Called function {inspect.stack()[1].function} for types {', '.join([str(type(elem)) for elem in types])}.\
                    Call function {other_function} instead."
        super().__init__(message)
