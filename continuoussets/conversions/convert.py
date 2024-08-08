from __future__ import annotations

# import numpy as np

from continuoussets.convexsets.interface_convexset import IConvexSet
from continuoussets.convexsets.interval import Interval
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.hpolyhedron import HPolyhedron
from continuoussets.conversions.interface_conversion import IConversion


# class for all containment checks
class Convert(IConversion):
    
    # constructor instantiates a Convert object
    # main task is to set the variables and decide which function to call for the evaluation
    def __init__(self, S: IConvexSet, set_class: str, *, mode: str = 'exact'):

        # call superclass constructor
        super().__init__(S, set_class = set_class, mode = mode)

        # select the correct function
        if isinstance(self.set, Interval):
            if self.set_class == 'Interval':
                self.fun = self._converts_interval_interval
            elif self.set_class == 'Zonotope':
                self.fun = self._converts_interval_zonotope
            elif self.set_class == 'VPolytope':
                self.fun = self._converts_interval_vpolytope
            elif self.set_class == 'HPolyhedron':
                self.fun = self._converts_interval_hpolyhedron

        elif isinstance(self.set, Zonotope):
            if self.set_class == 'Interval':
                self.fun = self._converts_zonotope_interval
            elif self.set_class == 'Zonotope':
                self.fun = self._converts_zonotope_zonotope
            elif self.set_class == 'VPolytope':
                self.fun = self._converts_zonotope_vpolytope
            elif self.set_class == 'HPolyhedron':
                self.fun = self._converts_zonotope_hpolyhedron

        elif isinstance(self.set, VPolytope):
            if self.set_class == 'Interval':
                self.fun = self._converts_vpolytope_interval
            elif self.set_class == 'Zonotope':
                self.fun = self._converts_vpolytope_zonotope
            elif self.set_class == 'VPolytope':
                self.fun = self._converts_vpolytope_vpolytope
            elif self.set_class == 'HPolyhedron':
                self.fun = self._converts_vpolytope_hpolyhedron

        elif isinstance(self.set, HPolyhedron):
            if self.set_class == 'Interval':
                self.fun = self._converts_hpolyhedron_interval
            elif self.set_class == 'Zonotope':
                self.fun = self._converts_hpolyhedron_zonotope
            elif self.set_class == 'VPolytope':
                self.fun = self._converts_hpolyhedron_vpolytope
            elif self.set_class == 'HPolyhedron':
                self.fun = self._converts_hpolyhedron_hpolyhedron

    # evaluate the containment problem
    def __call__(self) -> IConvexSet:
        return self.fun()

    # Interval contains Interval?
    def _converts_interval_interval(self) -> Interval:
        pass

    def _converts_interval_zonotope(self) -> Zonotope:
        pass

    def _converts_interval_vpolytope(self) -> VPolytope:
        pass

    def _converts_interval_hpolyhedron(self) -> HPolyhedron:
        pass

    def _converts_zonotope_interval(self) -> Interval:
        pass

    def _converts_zonotope_zonotope(self) -> Zonotope:
        pass

    def _converts_zonotope_vpolytope(self) -> VPolytope:
        pass

    def _converts_zonotope_hpolyhedron(self) -> HPolyhedron:
        pass

    def _converts_vpolytope_interval(self) -> Interval:
        pass

    def _converts_vpolytope_zonotope(self) -> Zonotope:
        pass

    def _converts_vpolytope_vpolytope(self) -> VPolytope:
        pass

    def _converts_vpolytope_hpolyhedron(self) -> HPolyhedron:
        pass

    def _converts_hpolyhedron_interval(self) -> Interval:
        pass

    def _converts_hpolyhedron_zonotope(self) -> Zonotope:
        pass

    def _converts_hpolyhedron_vpolytope(self) -> VPolytope:
        pass

    def _converts_hpolyhedron_hpolyhedron(self) -> HPolyhedron:
        pass
