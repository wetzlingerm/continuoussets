from __future__ import annotations

from abc import ABC
from typing import Union

import numpy as np

from continuoussets.convexsets.interface_convexset import IConvexSet

if __name__ == '__main__':
    print('This is the ISetOperation class')


# interface for set operations
class ISetOperation(ABC):

    def __init__(self,
                 first_operand: Union[IConvexSet, np.ndarray],
                 second_operand: Union[IConvexSet, np.ndarray],
                 *, mode = 'exact'):

        # assign sets
        self.first_operand = first_operand
        self.second_operand = second_operand

        # store mode
        self.mode = mode

        # set variable for function pointer
        self.fun = None
