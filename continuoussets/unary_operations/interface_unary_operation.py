from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

import numpy as np
from continuoussets.utils.auxiliary import SetPair

if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


if __name__ == '__main__':
    print('This is the IUnaryOperation class')


# interface for unary operations
class IUnaryOperation(ABC):

    def __init__(self,
                 S: Union['IConvexSet', np.ndarray],
                 set_class: str,
                 *args, **kwargs) -> IUnaryOperation:
        
        # assign sets
        self.first_operand = S
        self.set_class = set_class

        # store additional input arguments (mode, tolerances)
        self.args = args
        self.kwargs = kwargs

        # variable for function pointer to specific implementation
        # read out class names as strings (required since we would get full file structure otherwise)
        self.strategy_key = SetPair(self.first_operand, self.set_class)
        self.func = None

    # evaluate the operation (due to type checking in subclasses)
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
