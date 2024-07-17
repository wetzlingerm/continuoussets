from __future__ import annotations

from typing import Union

import numpy as np
# from scipy.optimize import linprog
from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils.exceptions import OtherFunctionError

if __name__ == '__main__':
    print('This is the Hpolyhedron class.')


class HPolyhedron(ConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *, A: Union[np.ndarray, list, float, int] = None,
                 b: Union[np.ndarray, list, float, int] = None, validate: bool = True):
        # some input has to be given
        if A is None or b is None:
            raise ValueError('HPolyhedron:__init__',
                             'No input arguments provided to constructor')

        # convert to numpy if possible
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        if not isinstance(b, np.ndarray):
            b = np.array(b)

        # validate input arguments
        if self.validate and validate:
            if b.ndim != 1:
                raise ValueError('HPolyhedron:__init__',
                                 'Offset must be a 1D array.')
            if A.ndim > 2:
                raise ValueError('HPolyhedron:__init__',
                                 'Constraint matrix must be a 1D or 2D array.')
            elif A.shape[0] != b.size:
                raise ValueError('HPolyhedron:__init__',
                                 'Dimension of constrained matrix and offset must match.')

        self.dimension = b.size
        self.A = A
        self.b = b

    # display
    def __repr__(self):
        """Representation on the command window.

        Returns:
            str: Description of the HPolyhedron object.
        """
        newline = '\n'
        return f'dimension: {self.dimension}{newline}'

    # translation by vector
    def __add__(self, other: np.ndarray) -> HPolyhedron:
        """Translation of a HPolyhedron by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If HPolyhedron + ConvexSet, call minkowski_sum instead.

        Returns:
            HPolyhedron: Result of the translation.
        """
        self._checkOtherOperand(other)
        
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return HPolyhedron(A = self.A, b = self.b - self.A*other, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
        
    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        raise NotImplementedError
    
    # unary minus
    def __neg__(self) -> HPolyhedron:
        """Unary minus operator.

        Returns:
            HPolyhedron: Input HPolyhedron times -1.
        """
        return HPolyhedron(A = -self.A, b = self.b, validate = False)

    # unary plus
    def __pos__(self) -> HPolyhedron:
        """Unary plus operator.

        Returns:
            HPolyhedron: Same as input HPolyhedron.
        """
        return HPolyhedron(A = self.A, b = self.b, validate = False)
    
    # translation by vector
    def __sub__(self, other: np.ndarray) -> HPolyhedron:
        """Translation of a HPolyhedron by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If HPolyhedron - ConvexSet, call minkowski_difference instead.

        Returns:
            HPolyhedron: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return HPolyhedron(A = self.A, b = self.b + self.A*other, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        self._checkOtherOperand(direction)

        raise NotImplementedError
    
    # check for boundedness
    def bounded(self) -> bool:

        # check if value of support function is finite in all directions of the nD simplex
        raise NotImplementedError

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # 1. convert other sets to Hpolyhedron
        # 2. block-concatenation
        raise NotImplementedError

    # center
    def center(self) -> np.ndarray:

        # LP for Chebyshev center (raise error for unbounded sets)
        raise NotImplementedError
    
    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> HPolyhedron:

        # check for redundancy of each halfspace constraint
        raise NotImplementedError

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        # np.ndarray: insert in to Ax <= b
        # ConvexSet: compute support function value along all normal vectors in A
        raise NotImplementedError

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)
        
        if mode == ['inner', 'exact']:
            raise NotImplementedError
        
        # 'outer': interval outer approximation of ConvexSet,
        #          add constraints in A with larger b (support function call for ConvexSet)
        raise NotImplementedError
    
    def empty(self) -> bool:
        
        # check emptiness using LP
        raise NotImplementedError
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)

        # LPs for all other ConvexSet classes
        raise NotImplementedError

    # conversion to interval
    def interval(self, *, mode: str = 'outer') -> dict:
        self._checkMode(mode)

        # support function in positive/negative axis-aligned directions
        # raise error if polyhedron is unbounded
        raise NotImplementedError
    
    # linear map
    def matmul(self, matrix: np.ndarray) -> HPolyhedron:
        self._checkMatrix(matrix)

        # case differentiation between square invertible matrices and projections
        raise NotImplementedError

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self + other
        
        # 1. convert other set to Hpolyhedron
        # 2. concatenation/lifting
        # 3. projection onto first n dimensions
        raise NotImplementedError
    
    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other

        # b vector - support function values along all normal vectors in A
        raise NotImplementedError
    
    # projection onto subspace
    def project(self, *, axis: tuple) -> HPolyhedron:
        self._checkSubspace(axis)

        # implement fourier_motzkin helper function
        raise NotImplementedError

    # representation by other set representation
    def represents(self, *, set_class: str) -> bool:
        self._checkSetClass(set_class)

        if set_class == 'HPolyhedron':
            return True
        
        # VPolytope: only if HPolyhedron is bounded
        # Interval: minimal representation only axis-aligned constraints (all dims!)
        # Zonotope: raise NotImplementedError
        
        raise NotImplementedError
    
    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        self._checkOtherOperand(direction)

        # LP for value and vector (simultaneously)
        raise NotImplementedError
        # return (value, vector)
    
    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of a HPolyhedron H.

        Returns:
            np.ndarray: 2D array containing vertices as columns.
        """
        # obtain minimal representation
        # H = self.compact()

        # could be difficult... some built-in function?
        raise NotImplementedError

    # volume
    def volume(self) -> float:
        raise NotImplementedError

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'outer') -> dict:
        self._checkMode(mode)

        # call conversion to interval and convert interval to zonotope
        raise NotImplementedError
        # return {'c': ..., 'G': ...}
