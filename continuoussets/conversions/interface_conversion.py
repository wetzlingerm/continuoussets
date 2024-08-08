from __future__ import annotations

from abc import ABC
from typing import Union

import numpy as np

from continuoussets.convexsets.interface_convexset import IConvexSet

if __name__ == '__main__':
    print('This is the IConversion class')


# interface for conversions
class IConversion(ABC):

    def __init__(self,
                 S: Union[IConvexSet, np.ndarray], *,
                 set_class: str, mode: str = 'exact'):
        
        # store set and class of target representation
        self.set = S
        self.set_class = set_class

        # store conversion mode
        self.mode = mode

        # set variable for function pointer
        self.fun = None
