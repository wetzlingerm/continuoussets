from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils.exceptions import OtherFunctionError

if __name__ == '__main__':
    print('This is the VPolytope class.')


class VPolytope(ConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *, V: Union[np.ndarray, list, float, int] = None, validate: bool = True):
        # enforce that some vertices are given
        if V is None:
            raise ValueError('VPolytope:__init__',
                             'No input arguments provided to constructor')

        # convert to numpy if possible
        if not isinstance(V, np.ndarray):
            V = np.array(V)

        self.dimension = V.shape[0]
        self.V = V

    # display
    def __repr__(self):
        """Representation on the command window.

        Returns:
            str: Description of the VPolytope object.
        """
        newline = '\n'
        return f'dimension: {self.dimension}{newline}'

    # translation by vector
    def __add__(self, other: np.ndarray) -> VPolytope:
        """Translation of a VPolytope by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If other is a ConvexSet, call minkowski_sum instead.

        Returns:
            VPolytope: Result of the translation.
        """
        self._checkOtherOperand(other)
        
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return VPolytope(V = self.V + other, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
        
    
    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        raise NotImplementedError
    
    # unary minus
    def __neg__(self) -> VPolytope:
        """Unary minus operator.

        Returns:
            VPolytope: Input VPolytope times -1.
        """
        return VPolytope(V = -self.V, validate = False)

    # unary plus
    def __pos__(self) -> VPolytope:
        """Unary plus operator.

        Returns:
            VPolytope: Same as input VPolytope.
        """
        return VPolytope(V = self.V, validate = False)
    
    # translation by vector
    def __sub__(self, other: np.ndarray) -> VPolytope:
        """Translation of a VPolytope by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If VPolytope - ConvexSet, call minkowski_difference instead.

        Returns:
            VPolytope: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return VPolytope(V = self.V - other, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        self._checkOtherOperand(direction)

        raise NotImplementedError

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        raise NotImplementedError

    # center
    def center(self) -> np.ndarray:
        # compute some approximation?
        raise NotImplementedError
    
    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> VPolytope:
        """Minimal representation of a VPolytope VP.
        Removes points that are inside of the convex hull of all points.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-12.

        Returns:
            VPolytope: VPolytope in minimal representation.
        """
        return VPolytope(V = self.V[ConvexHull(self.V).vertices, :], validate = False)

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        raise NotImplementedError

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        raise NotImplementedError
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)

        raise NotImplementedError

    # conversion to interval
    def interval(self, *, mode: str = 'outer') -> dict:
        self._checkMode(mode)

        # take minimum and maximum in every dimension
        lower_bound = np.min(self.V, axis = 0)
        upper_bound = np.max(self.V, axis = 0)

        return {'lb': lower_bound, 'ub': upper_bound}
    
    # linear map
    def matmul(self, matrix: np.ndarray) -> VPolytope:
        """Linear map of a VPolytope VP by a matrix (np.ndarray) M.
        Defined as {M s | s in VP}.

        Args:
            matrix (np.ndarray): Matrix for left-multiplication.

        Returns:
            VPolytope: Result of the matrix multiplication.
        """
        self._checkMatrix(matrix)

        # simple linear transformation of all points
        return VPolytope(V = np.matmul(matrix, self.V), validate = False)

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self + other

        # compute vertices of other set and add each combination
        raise NotImplementedError
    
    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other

        raise NotImplementedError
    
    # projection onto subspace
    def project(self, *, axis: tuple) -> VPolytope:
        """Projection of a VPolytope VP onto a subspace.

        Args:
            axis (tuple): Subspace for projection.

        Returns:
            VPolytope: Projected VPolytope.
        """
        self._checkSubspace(axis)

        # convert tuples to lists for indexing
        return VPolytope(V = self.V[list(axis), :], validate = False)

    # representation by other set representation
    def represents(self, *, set_class: str) -> bool:
        self._checkSetClass(set_class)

        if set_class == ['VPolytope', 'HPolyhedron']:
            return True
        
        # Zonotope, Interval?
        raise NotImplementedError
    
    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        self._checkOtherOperand(direction)

        # value = max_i np.dot(direction, v_i)
        # vector = argmax_i np.dot(direction, v_i)
        # return (value, vector)
        raise NotImplementedError
    
    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of a VPolytope VP.

        Returns:
            np.ndarray: 2D array containing vertices as columns.
        """
        # return minimal representation
        return self.compact().V

    # volume
    def volume(self) -> float:
        raise NotImplementedError

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'outer') -> dict:
        self._checkMode(mode)

        # convert to interval
        interval_dict = self.interval(mode = 'outer')
        lower_bound = interval_dict['lb']
        upper_bound = interval_dict['ub']
        # convert interval to zonotope (note: we cannot call Interval methods here)
        center = (upper_bound + lower_bound)/2
        generators = np.diag((upper_bound - lower_bound)/2)
        
        return {'c': center, 'G': 0.5*generators[:, ~np.all(generators == 0, axis=0)]}
