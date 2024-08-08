from __future__ import annotations

from typing import Union

import numpy as np

from continuoussets.convexsets.interface_convexset import IConvexSet
from continuoussets.convexsets.interval import Interval
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.hpolyhedron import HPolyhedron
from .interface_predicate import IPredicate


# class for all containment checks
class Equals(IPredicate):
    
    # constructor instantiates a Equals object
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S1: IConvexSet, S2: Union[IConvexSet, np.ndarray], *,
                 rtol: float = 1e-8, atol: float = 1e-10):

        # call superclass constructor
        super().__init__(S1, S2, rtol = rtol, atol = atol)

        # select the correct function
        if isinstance(self.first_operand, Interval):
            if isinstance(self.second_operand, Interval):
                self.fun = self._equals_interval_interval
            elif isinstance(self.second_operand, Zonotope):
                self.fun = self._equals_interval_zonotope
            elif isinstance(self.second_operand, VPolytope):
                self.fun = self._equals_interval_vpolytope
            elif isinstance(self.second_operand, HPolyhedron):
                self.fun = self._equals_interval_hpolyhedron

        elif isinstance(self.first_operand, Zonotope):
            if isinstance(self.second_operand, Interval):
                self.fun = self._equals_zonotope_interval
            elif isinstance(self.second_operand, Zonotope):
                self.fun = self._equals_zonotope_zonotope
            elif isinstance(self.second_operand, VPolytope):
                self.fun = self._equals_zonotope_vpolytope
            elif isinstance(self.second_operand, HPolyhedron):
                self.fun = self._equals_zonotope_hpolyhedron

        elif isinstance(self.first_operand, VPolytope):
            if isinstance(self.second_operand, Interval):
                self.fun = self._equals_vpolytope_interval
            elif isinstance(self.second_operand, Zonotope):
                self.fun = self._equals_vpolytope_zonotope
            elif isinstance(self.second_operand, VPolytope):
                self.fun = self._equals_vpolytope_vpolytope
            elif isinstance(self.second_operand, HPolyhedron):
                self.fun = self._equals_vpolytope_hpolyhedron

        elif isinstance(self.first_operand, HPolyhedron):
            if isinstance(self.second_operand, Interval):
                self.fun = self._equals_hpolyhedron_interval
            elif isinstance(self.second_operand, Zonotope):
                self.fun = self._equals_hpolyhedron_zonotope
            elif isinstance(self.second_operand, VPolytope):
                self.fun = self._equals_hpolyhedron_vpolytope
            elif isinstance(self.second_operand, HPolyhedron):
                self.fun = self._equals_hpolyhedron_hpolyhedron

    # evaluate the containment problem
    def __call__(self) -> bool:
        return self.fun()

    # Interval equals Interval?
    def _equals_interval_interval(self) -> bool:
        pass

    def _equals_interval_zonotope(self) -> bool:
        pass

    def _equals_interval_vpolytope(self) -> bool:
        pass

    def _equals_interval_hpolyhedron(self) -> bool:
        pass

    def _equals_zonotope_interval(self) -> bool:
        pass

    def _equals_zonotope_zonotope(self) -> bool:
        pass

    def _equals_zonotope_vpolytope(self) -> bool:
        pass

    def _equals_zonotope_hpolyhedron(self) -> bool:
        pass

    def _equals_vpolytope_interval(self) -> bool:
        pass

    def _equals_vpolytope_zonotope(self) -> bool:
        pass

    def _equals_vpolytope_vpolytope(self) -> bool:
        pass

    def _equals_vpolytope_hpolyhedron(self) -> bool:
        pass

    def _equals_hpolyhedron_interval(self) -> bool:
        pass

    def _equals_hpolyhedron_zonotope(self) -> bool:
        pass

    def _equals_hpolyhedron_vpolytope(self) -> bool:
        pass

    def _equals_hpolyhedron_hpolyhedron(self) -> bool:
        pass
