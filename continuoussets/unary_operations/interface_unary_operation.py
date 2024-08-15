from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable, TYPE_CHECKING

import numpy as np
from continuoussets.utils.auxiliary import SetPair

if TYPE_CHECKING:
    from continuoussets.convexsets.interface_convexset import IConvexSet


if __name__ == '__main__':
    print('This is the IUnaryOperation class')


# interface for unary operations
class IUnaryOperation(ABC):

    # class variable for selection of strategies
    strategies: Dict[Tuple[str, str], Callable] = dict()

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
        self.func = None

    @classmethod
    def register_strategy(cls: IUnaryOperation, pair: Union[SetPair, Tuple[SetPair]]) -> Callable:
        def decorator(func: Callable):
            if isinstance(pair, Tuple):
                [cls.strategies.update({i_pair: func}) for i_pair in pair]
            else:
                cls.strategies.update({pair: func})
            return func
        return decorator

    # evaluate the operation (due to type checking in subclasses)
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    # read out class names as strings (required since we would get full file structure otherwise)
    def get_strategy_key(self: IUnaryOperation) -> SetPair:
        return SetPair(self.first_operand.__class__.__name__, self.set_class)
