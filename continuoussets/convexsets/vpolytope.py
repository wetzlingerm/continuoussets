from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError

if __name__ == '__main__':
    print('This is the VPolytope class.')


class VPolytope(ConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *, V: Union[np.ndarray, list, float, int] = None, validate: bool = True):
        if self.validate and validate:
            # enforce that some vertices are given
            if V is None:
                raise ValueError('VPolytope:__init__',
                                 'No input arguments provided to constructor')
            # check correct type
            elif (not isinstance(V, int) and not isinstance(V, float)
                  and not isinstance(V, list) and not isinstance(V, np.ndarray)):
                raise TypeError('VPolytope:__init__',
                                'Vertices must be int, float, list or np.ndarray')

        # convert to np.ndarray
        if not isinstance(V, np.ndarray):
            if isinstance(V, int) or isinstance(V, float):
                V = np.array([[float(V)]])
            elif isinstance(V, list):
                if not all(isinstance(element, float) for element in V):
                    V = [float(element) for element in V]
                V = np.array(V)

        # expand to 2D array
        if V.ndim == 1:
            V = np.reshape(V, (1, V.size))

        # post-check: no higher than 2D
        if self.validate and validate:
            if V.ndim > 2:
                raise ValueError('VPolytope:__init__',
                                 'V must be 1D or 2D.')

        self.dimension = V.shape[0]
        self.V = V.copy()

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
            return VPolytope(V = self.V + other.reshape(self.dimension, 1), validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
        
    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = 'exact'))
        
        # compute minimal representation of both sets and compare list of vertices
        other_minimal = other.compact()
        self = self.compact()
        return comparison.compare_matrices(self.V, other_minimal.V)
    
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
            return VPolytope(V = self.V - other.reshape(self.dimension, 1), validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        self._checkOtherOperand(direction)

        # todo
        raise NotImplementedError

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # convert other set to VPolytope
        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = mode))

        # all potential combinations of vertices
        V_product = np.vstack((np.tile(self.V, (1, other.V.shape[1])), np.repeat(other.V, self.V.shape[1], axis = 1)))
        return VPolytope(V = V_product, validate = False)

    # center
    def center(self) -> np.ndarray:
        # trivial solution for single vertex
        if (self.V.shape[1] == 1):
            return self.V

        # todo: weight each vertex by same factor and compute that 'center' (guaranteed to be contained in vpolytope)
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
        # for ConvexHull function, we need at least n+1 vertices
        if self.V.shape[1] == 1:
            # single vertex
            return VPolytope(V = self.V, validate = False)
        
        elif self.V.shape[1] <= self.dimension:
            # check manually for duplicates
            index_nonduplicate = np.ones(self.V.shape[1], dtype = bool)
            for j in range(self.V.shape[1]):
                other_vertices = np.hstack((self.V[:, 0:j], self.V[:, j+1:-1]))
                this_vertex = np.reshape(self.V[:, j], (self.dimension, 1))
                if np.any(np.isclose(np.linalg.norm(other_vertices - this_vertex), 0, rtol=rtol)):
                    index_nonduplicate[j] = False
            # remove duplicates
            return VPolytope(V = self.V[:, index_nonduplicate], validate = False)

        else:
            # note: ConvexHull expects vertices as rows
            return VPolytope(V = self.V[:, ConvexHull(self.V.T).vertices], validate = False)

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self._contains_point(other)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(), mode = 'exact')

        return self == other.convex_hull(self)

    def _contains_point(self, other: np.ndarray) -> bool:
        # no checks in underscore-functions
        other = np.reshape(other, (self.dimension, 1))

        # number of vertices in the outer body
        number_vertices = self.V.shape[1]

        # objective function
        c = np.zeros(number_vertices)

        # constraints
        A_eq = np.vstack((self.V, np.ones((1, number_vertices))))
        b_eq = np.vstack((other, 1))
        A_ub = -np.eye(number_vertices)
        b_ub = np.zeros((number_vertices, 1))

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # vector is contained if the LP is feasible
        return res.success

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> VPolytope:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = mode))

        V_all = np.hstack((self.V, other.V))
        return VPolytope(V = V_all, validate = False)
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)

        # todo: linprogs...
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
        elif not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(), mode = mode)

        # todo: add each combination
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
        
        # todo: Zonotope, Interval?
        raise NotImplementedError
    
    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Support function evaluation of a VPolytope VP in a direction d.
        Value defined as max_{s in VP} d^T * s.
        Vector defined as arg max_{s in VP} d^T * s.

        Args:
            direction (np.ndarray): Direction along which to evaluate the support function.

        Returns:
            tuple[float, np.ndarray]: Support value and support vector.
        """
        self._checkOtherOperand(direction)

        dot_product = direction @ self.V
        value = np.max(dot_product)
        max_index = np.flatnonzero(dot_product == value)[0]
        vector = np.reshape(self.V[:, max_index], (self.dimension, 1))
        return (value, vector)
    
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
        """Volume computation of a Zonotope Z.
        Note: Degenerate sets have a volume of zero.

        Returns:
            float: Volume.
        """
        # degenerate polytopes have volume 0
        if self.V.shape[1] == 1:
            return 0
        # todo: check other degenerate cases

        raise NotImplementedError
    
    # conversion to vpolytope
    def vpolytope(self, *, mode: str = 'exact') -> dict:
        """Overloaded conversion to VPolytope.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an VPolytope object.
        """
        self._checkMode(mode)

        return {'V': self.V}

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'outer') -> dict:
        """Conversion to Zonotope. We use a box enclosure.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an Zonotope object.
        """
        self._checkMode(mode)

        # convert to interval
        interval_dict = self.interval(mode = 'outer')
        lower_bound = interval_dict['lb']
        upper_bound = interval_dict['ub']
        # convert interval to zonotope (note: we cannot call Interval methods here)
        center = (upper_bound + lower_bound)/2
        generators = np.diag((upper_bound - lower_bound)/2)
        generators = 0.5*generators[:, ~np.all(generators == 0, axis=0)]
        
        return {'c': center, 'G': generators}
