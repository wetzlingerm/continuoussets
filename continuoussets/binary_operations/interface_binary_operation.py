from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Callable, TYPE_CHECKING

import numpy as np
from continuoussets.utils.auxiliary import SetPair

if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


if __name__ == '__main__':
    print('This is the IBinaryOperation class')


# interface for binary operations
class IBinaryOperation(ABC):

    # does the order of operands matter?
    ordered = True

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
        # read out class names as strings (required since we would get full file structure otherwise)
        self.strategy_key = SetPair(self.first_operand, self.second_operand)
        self.func: Callable = None

    # evaluate the operation (due to type checking in subclasses)
    @abstractmethod
    def __call__(self):
        pass
    
    def select_binary_strategy(self: IBinaryOperation) -> IBinaryOperation:
        # read out strategy for given ordering
        self.func = self.select_strategy(self.strategy_key)
        if self.func is None:
            if self.ordered:
                raise NotImplementedError
            
            self.strategy_key = SetPair(self.second_operand, self.first_operand)
            self.func = self.select_strategy(self.strategy_key)
            if self.func is None:
                raise NotImplementedError
            
            # re-order operands for call of concrete implementation in subclass
            helper = self.first_operand
            self.first_operand = self.second_operand
            self.second_operand = helper

        return self
