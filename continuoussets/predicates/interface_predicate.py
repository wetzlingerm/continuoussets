from __future__ import annotations

from abc import ABC
from typing import Union

import numpy as np

from continuoussets.convexsets.interface_convexset import IConvexSet


if __name__ == '__main__':
    print('This is the IPredicate class')


# interface for predicates
class IPredicate(ABC):

    def __init__(self,
                 first_operand: Union[IConvexSet, np.ndarray],
                 second_operand: Union[IConvexSet, np.ndarray],
                 *args, **kwargs):
        
        # assign sets
        self.first_operand = first_operand
        self.second_operand = second_operand

        # store additional input arguments
        self.args = args
        self.kwargs = kwargs

        # set variable for function pointer
        self.fun = None
