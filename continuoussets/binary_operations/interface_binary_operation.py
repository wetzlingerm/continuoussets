from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


if __name__ == '__main__':
    print('This is the IBinaryOperation class')


# pair of sets
class SetPair:

    def __init__(self, first_operand: str, second_operand: str, *, ordered = True) -> SetPair:
        # save operands and ordering
        self.first_operand = first_operand
        self.second_operand = second_operand
        self.ordered = ordered

    def __repr__(self) -> str:
        newline = '\n'
        return f"first operand: {self.first_operand}{newline}" \
               f"second operand: {self.first_operand}{newline}" \
               f"ordering: {self.ordered}"
        
    def __eq__(self, other: SetPair) -> bool:
        # check for equality, may depend on order
        if not isinstance(other, SetPair):
            return False
        elif self.ordered:
            return self.first_operand == other.first_operand and self.second_operand == other.second_operand
        else:
            return (self.first_operand == other.first_operand and self.second_operand == other.second_operand) \
                    or (self.first_operand == other.second_operand and self.second_operand == other.first_operand)
        
    def __hash__(self):
        return hash(self.first_operand) + hash(self.second_operand)


# interface for predicates
class IBinaryOperation(ABC):

    # class variable for selection of strategies
    strategies: Dict[Tuple[str, str], Callable] = dict()
    # is ordering relevant?
    ordered_operation = True

    def __init__(self,
                 first_operand: Union['IConvexSet', np.ndarray],
                 second_operand: Union['IConvexSet', np.ndarray],
                 *args, **kwargs) -> IBinaryOperation:
        
        # assign sets
        self.first_operand = first_operand
        self.second_operand = second_operand

        # store additional input arguments
        self.args = args
        self.kwargs = kwargs

        # variable for function pointer to specific implementation
        self.func = None

    @classmethod
    def register_strategy(cls: IBinaryOperation, pair: SetPair) -> Callable:
        def decorator(func: Callable):
            cls.strategies[pair] = func
            return func
        return decorator

    # evaluate the predicate (due to type checking in subclasses)
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    # read out class names as strings (required since we would get full file structure otherwise)
    # additionally, let the pair know whether the ordering is important
    def get_strategy_key(self: IBinaryOperation) -> SetPair:
        return SetPair(self.first_operand.__class__.__name__,
                       self.second_operand.__class__.__name__,
                       ordered = self.ordered_operation)
