from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog

from continuoussets.convexsets.interface_convexset import IConvexSet
from continuoussets.utils.exceptions import OtherFunctionError
from continuoussets.utils.auxiliary import number_singular_values, convex_hull

if __name__ == '__main__':
    print('This is the VPolytope class.')


class VPolytope(IConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *,
                 V: Union[np.ndarray, list, float, int] = None,
                 validate: bool = True):
        """Instantiates a VPolytope object VP = {sum_i v_i beta_i | sum_i beta_i = 1, beta >= 0}.

        Args:
            V (Union[np.ndarray, list, float, int], optional): Points. Defaults to None.
            validate (bool, optional): Input argument check. Defaults to True.

        Raises:
            ValueError: Vertices must be provided.
            TypeError: Vertices must be int, float, list or np.ndarray.
            ValueError: Vertices array must be 1D or 2D.

        Returns:
            VPolytope: Polytope.
        """
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
                                 'Vertices array must be 1D or 2D.')

        self.dimension = V.shape[1]
        self.V = V.copy()

    # deep copy
    def copy(self) -> VPolytope:
        """Returns a deep copy of an VPolytope.

        Returns:
            VPolytope: Copied VPolytope.
        """
        return VPolytope(V = self.V.copy(), validate = False)

    # display
    def __repr__(self):
        """Representation on the command window.

        Returns:
            str: Description of the VPolytope object.
        """
        newline = '\n'
        return f'dimension: {self.dimension}{newline}V: {self.V}'

    # translation by vector
    def __add__(self, other: np.ndarray) -> VPolytope:
        """Translation of a VPolytope by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If other is a IConvexSet, call minkowski_sum instead.

        Returns:
            VPolytope: Result of the translation.
        """
        self._checkOtherOperand(other)
        
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return VPolytope(V = self.V + other, validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
    
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
            OtherFunctionError: If VPolytope - IConvexSet, call minkowski_difference instead.

        Returns:
            VPolytope: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return VPolytope(V = self.V - other, validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')
        
    # basis of the affine hull (for degenerate sets)
    def basis_affine_hull(self) -> tuple:
        """Computes a basis of the affine hull of a VPolytope VP.

        Returns:
            tuple: Matrix with basis vectors, number of basis vectors.
        """
        # ensure that the origin is contained
        V_shifted = self.V - self.center()
        # compute singular value decomposition
        matrix, S, _ = np.linalg.svd(V_shifted.T)
        # check number of singular values (with built-in tolerance)
        s = number_singular_values(S)
        return (matrix, s)

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of an VPolytope VP in a given direction.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Raises:
            NotImplementedError: VPolytope must be non-degenerate.

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        if self.degenerate():
            raise NotImplementedError

        # LP formulation for boundary point computation
        # min_{beta,x,l}    -l
        # s.t.              V * beta - x = 0
        #                   - x + l*dir = 0
        #                   sum beta = 1
        #                   -beta <= 0

        # retreive information
        n, m = self.dimension, self.number_vertices()

        # objective function
        c = np.hstack((np.zeros(m+n), -1.))

        # equality constraints
        A_eq = np.block([[self.V.T, -np.eye(n), np.zeros((n, 1))],
                         [np.zeros((n, m)), -np.eye(n), np.reshape(direction, (n, 1))],
                         [np.ones((1, m)), np.zeros((1, n)), 0.]])
        b_eq = np.hstack((np.zeros(2*n), 1.))
        A_ub = np.hstack((-np.eye(m), np.zeros((m, n+1))))
        b_ub = np.zeros(m)

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))
        # infeasible means there is not point along the given direction
        if res.status == 2:
            return None
        return -res.fun * direction
    
    # boundedness
    def bounded(self) -> bool:
        """Checks if a VPolytope VP is bounded.

        Returns:
            bool: Boundedness.
        """
        return True

    # center
    def center(self) -> np.ndarray:
        """Center of the VPolytope VP.
        Defined as sum_i 1/N*v_i, where N is the number of vertices.
        Note: This is no more than a good guess, however, it is guaranteed to be in the interior if an interior exists.

        Returns:
            np.ndarray: Center.
        """
        # weight each point by the same factor -> guaranteed to be contained in vpolytope
        return np.sum(1/self.number_vertices()*self.V, axis = 0)
    
    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> VPolytope:
        """Minimal representation of a VPolytope VP.
        Removes points that are inside of the convex hull of all points.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-12.

        Returns:
            VPolytope: VPolytope in minimal representation.
        """
        V_minimal = convex_hull(self.V)
        return VPolytope(V = V_minimal, validate = False)        
    
    # degeneracy
    def degenerate(self, *, tol: float = 1e-12) -> bool:
        """Checks if a VPolytope VP is degenerate.

        Args:
            tol (float, optional): Tolerance. Defaults to 1e-12.

        Returns:
            bool: Degeneracy.
        """
        return np.linalg.matrix_rank(self.V - np.mean(self.V, axis = 0), tol = tol) < self.dimension

    # emptiness
    def empty(self) -> bool:
        """Checks if a VPolytope VP is empty.

        Returns:
            bool: Emptiness.
        """
        return False
    
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
        return VPolytope(V = np.matmul(self.V, matrix.T), validate = False)
    
    # number of vertices
    def number_vertices(self) -> int:
        """Returns the number of vertices of a VPolytope VP.

        Returns:
            int: Number of vertices.
        """
        return self.V.shape[0]
    
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
        return VPolytope(V = self.V[:, list(axis)], validate = False)

    # projection onto its own affine hull
    def project_affine_hull(self) -> tuple:
        """Projects a VPolytope onto its own affine hull.
        For a degenerate VPolytope, the resulting VPolytope is of lower dimension, but non-degenerate.

        Returns:
            tuple: Projected VPolytope, projection matrix, center of new coordinate system in old coordinate system.
        """
        # compute basis of affine hull
        n, c = self.dimension, self.center()
        VP_shifted = self - c
        M_proj, r = VP_shifted.basis_affine_hull()
        # early exit if basis of affine hull is n-dimensional identity
        if r == n:
            return (self, np.eye(n), np.zeros(n))

        # map vertices onto lower dimensional space
        V_proj = np.matmul(M_proj.T, VP_shifted.V.T)
        V_proj = V_proj[np.invert(np.all(np.isclose(V_proj, 0.), axis = 1)), :]
        VP_proj = VPolytope(V = V_proj.T)
        return (VP_proj, M_proj, c)

    # reduction of set representation size
    def reduce(self, *, order: int) -> VPolytope:
        """Reduction of the set representation size of a VPolytope VP.

        Args:
            order (int): Reduced number of vertices.

        Raises:
            NotImplementedError: Currently not supported.

        Returns:
            VPolytope: VPolytope with reduced set representation size.
        """
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

        dot_product = np.matmul(self.V, direction)
        value = np.max(dot_product)
        max_index = np.flatnonzero(dot_product == value)[0]
        vector = self.V[max_index, :]
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
        if self.number_vertices() <= 1 or self.degenerate():
            return 0
        
        # computation for non-degenerate sets is not supported
        raise NotImplementedError
