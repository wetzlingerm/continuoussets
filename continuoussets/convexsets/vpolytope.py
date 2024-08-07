from __future__ import annotations

from typing import Union
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.linalg import svd
from pypoman import compute_polytope_halfspaces

from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError, ExactEvaluationImpossibleError
from continuoussets.utils.auxiliary import halfspace_representation_from_vector, \
                                           remove_duplicate_points, \
                                           number_singular_values

if __name__ == '__main__':
    print('This is the VPolytope class.')


class VPolytope(ConvexSet):
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
    def __eq__(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Set equality of a VPolytope VP with another set or vector S.
        Defined as forall i in VP: i in VP and forall s in S: s in VP?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Set equality.
        """
        self._checkOtherOperand(other)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = 'exact'))
        
        # compute minimal representation of both sets and compare list of vertices
        self, other_minimal = self.compact(), other.compact()
        return comparison.compare_matrices(self.V, other_minimal.V, rtol = rtol, atol = atol)
    
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
        
    # basis of the affine hull (for degenerate sets)
    def basis_affine_hull(self) -> np.ndarray:
        """Computes a basis of the affine hull of a VPolytope VP.

        Returns:
            np.ndarray: Matrix with basis vectors.
        """
        # ensure that the origin is contained
        V_shifted = self.V - self.center()
        # compute singular value decomposition
        matrix, S, _ = np.linalg.svd(V_shifted.T)
        # check number of singular values (with built-in tolerance)
        if number_singular_values(S) == self.dimension:
            return np.eye(self.dimension)
        return matrix

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of an VPolytope VP in a given direction.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Raises:
            NotImplementedError: VPolytope must contain the origin.
            NotImplementedError: VPolytope must be non-degenerate.

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        if self.degenerate():
            raise NotImplementedError
        elif not self.contains(np.zeros(self.dimension)):
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
        return -res.fun * direction
    
    # boundedness
    def bounded(self) -> bool:
        """Checks if a VPolytope VP is bounded.

        Returns:
            bool: Boundedness.
        """
        return True

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        """Cartesian product of a VPolytope VP and another set or vector S.
        Defined as {[a^T s^T]^T | a in VP, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            VPolytope: Result of the Cartesian product.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # convert other set to VPolytope
        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = mode))

        # all potential combinations of vertices
        V_product = np.hstack((np.tile(self.V, (other.number_vertices(), 1)),
                               np.repeat(other.V, self.number_vertices(), axis = 0)))
        return VPolytope(V = V_product, validate = False)

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
        # for ConvexHull function, we need at least n+1 vertices
        if self.number_vertices() == 1:
            # single vertex
            return VPolytope(V = self.V, validate = False)
        
        elif self.dimension == 1:
            # just take min and max
            return VPolytope(V = np.vstack((np.min(self.V, axis=0), np.max(self.V, axis=0))), validate = False)
        
        elif self.number_vertices() <= self.dimension:
            # check manually for duplicates
            index_nonduplicate = np.ones(self.number_vertices(), dtype = bool)
            for j in range(self.number_vertices()):
                other_vertices = np.vstack((self.V[0:j, :], self.V[j+1:-1, :]))
                this_vertex = self.V[j, :]
                if np.any(np.isclose(np.linalg.norm(other_vertices - this_vertex), 0, rtol = rtol)):
                    index_nonduplicate[j] = False
            # remove duplicates
            return VPolytope(V = self.V[index_nonduplicate, :], validate = False)

        else:
            # note: ConvexHull expects vertices as rows
            return VPolytope(V = self.V[ConvexHull(self.V).vertices, :], validate = False)

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks containment of a ConvexSet or vector (np.ndarray) S in a VPolytope VP.
        Defined as forall s in S: s in VP?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Containment.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self._contains_point(other, rtol = rtol, atol = atol)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = 'exact'))

        return self.__eq__(other.convex_hull(self), rtol = rtol, atol = atol)

    def _contains_point(self, other: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Point-in-VPolytope check.

        Args:
            other (np.ndarray): Vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Containment.
        """
        # no checks in underscore-functions
        # todo: use dual and integrate tolerances...

        # objective function
        c = np.zeros(self.number_vertices())

        # constraints
        A_eq = np.vstack((self.V.T, np.ones((1, self.number_vertices()))))
        b_eq = np.hstack((other, 1))
        A_ub = -np.eye(self.number_vertices())
        b_ub = np.zeros((self.number_vertices(), 1))

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # vector is contained if the LP is feasible
        return res.success

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        """Convex hull of a VPolytope VP and another set or vector S.
        Defined as {lambda*v + (1-lambda)*s | v in VP, s in S, lambda in [0,1]}

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            VPolytope: Result of the convex hull.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = mode))

        V_all = np.vstack((self.V, other.V))
        return VPolytope(V = V_all, validate = False)
    
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
    
    # conversion to hpolyhedron
    def hpolyhedron(self, *, mode: str = 'exact') -> dict:
        """Conversion of an VPolytope VP to an HPolyhedron HP.

        Args:
            mode (str, optional): Approximation of conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: keyword arguments for instantiation of a HPolyhedron object
        """
        self._checkMode(mode)

        n = self.dimension
        if self.number_vertices() == 1:
            A, b = halfspace_representation_from_vector(np.reshape(self.V, (n, )))
            return {'A': A, 'b': b}
        
        # shift vertices by mean
        center = np.mean(self.V, axis = 0)
        V = self.V - center

        # if polytope is degenerate, we need projection to affine hull and back
        U_, S_, V_ = svd(V.T)
        subspace_dimension = n - np.sum(np.isclose(S_, 0.))
        if subspace_dimension < n:
            # project vertices onto basis and filter out the subspace
            V = np.matmul(U_.T, V.T)
            V = V[0:subspace_dimension].T

        A, b = compute_polytope_halfspaces(V)

        if subspace_dimension < n:
            # project back to original dimension
            h = A.shape[0]
            A = np.vstack((np.hstack((A, np.zeros((h, (n-subspace_dimension))))),
                           np.hstack((np.zeros((2*(n-subspace_dimension), subspace_dimension)),
                                      np.vstack((np.eye(n-subspace_dimension), -np.eye(n-subspace_dimension)))))))
            b = np.hstack((b, np.zeros(2*(n - subspace_dimension))))
            A = np.matmul(A, U_.T)
        b += np.matmul(A, center)
        
        return {'A': A, 'b': b}
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if an VPolytope VP intersects another set of vector S.
        Defined as exists s in VP: s in S?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other, rtol = rtol, atol = atol)
        elif isinstance(other, VPolytope):
            return self._intersects_vpolytope(other, rtol = rtol, atol = atol)
        if type(other).__name__ == 'HPolyhedron':
            return other.intersects(self, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'Zonotope':
            return self._intersects_zonotope(other, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'Interval':
            return self._intersects_interval(other, rtol = rtol, atol = atol)
    
    # intersection check with vpolytope
    def _intersects_vpolytope(self, other: VPolytope, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope intersects another VPolytope.

        Args:
            other (VPolytope): VPolytope.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # read out number of vertices
        m1, m2 = self.number_vertices(), other.number_vertices()
        if (m1 == 1):
            return other.contains(self.V[0])
        elif (m2 == 1):
            return self.contains(other.V[0])

        # objective function
        c = np.zeros(m1 + m2)

        # constraints
        A_eq = np.vstack((np.hstack((self.V.T, -other.V.T)),
                          np.hstack((np.ones(m1), np.zeros(m2))),
                          np.hstack((np.zeros(m1), np.ones(m2)))))
        b_eq = np.hstack((np.zeros(self.dimension), np.array([1., 1.])))
        A_ub = -np.eye(m1 + m2)
        b_ub = np.zeros(m1 + m2)

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # vector is contained if the LP is feasible
        return res.success
    
    # intersection check with interval
    def _intersects_interval(self, other, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope intersects an Interval.

        Args:
            other (VPolytope): Interval.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # read out dimension and number of vertices
        n, m = self.dimension, self.number_vertices()

        # convert interval to halfspace representation
        hpolyhedron_dict = other.hpolyhedron()
        A, b = hpolyhedron_dict['A'], hpolyhedron_dict['b']

        # objective function
        c = np.zeros(m + n)

        # constraints
        A_eq = np.vstack((np.hstack((self.V.T, -np.eye(n))),
                          np.hstack((np.ones((1, m)), np.zeros((1, n))))))
        b_eq = np.hstack((np.zeros(n), 1.))
        A_ub = np.vstack((np.hstack((-np.eye(m), np.zeros((m, n)))),
                          np.hstack((np.zeros((2*n, m)), A))))
        b_ub = np.hstack((np.zeros(m), b))

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # vector is contained if the LP is feasible
        return res.success
    
    # intersection check with zonotope
    def _intersects_zonotope(self, other, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope intersects a Zonotope.

        Args:
            other (VPolytope): Zonotope.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # read out dimension, number of vertices, number of generators
        m, g = self.number_vertices(), other.number_generators()

        # objective function
        c = np.zeros(g + m)

        # constraints
        A_eq = np.vstack((np.hstack((other.G.T, -self.V.T)),
                          np.hstack((np.zeros((1, g)), np.ones((1, m))))))
        b_eq = np.hstack((-other.c, 1.))
        A_ub = np.vstack((np.hstack((np.eye(g), np.zeros((g, m)))),
                          np.hstack((-np.eye(g), np.zeros((g, m)))),
                          np.hstack((np.zeros((m, g)), -np.eye(m)))))
        b_ub = np.hstack((np.ones(2*g), np.zeros(m)))

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # vector is contained if the LP is feasible
        return res.success

    # conversion to interval
    def interval(self, *, mode: str = 'exact') -> dict:
        """Conversion to Interval.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Conversion to inner approximation not supported.
            ExactEvaluationImpossibleError: Exact conversion only possible in special cases.

        Returns:
            dict: Keyword arguments for instantiation of an Interval object.
        """
        self._checkMode(mode)

        if mode == 'inner':
            if not self.represents(set_class = 'Interval'):
                # exact evaluation is also an inner approximation
                raise NotImplementedError
        elif mode == 'exact':
            # only continue if polytope is actually an interval
            if not self.represents(set_class = 'Interval'):
                raise ExactEvaluationImpossibleError

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
        return VPolytope(V = np.matmul(self.V, matrix.T), validate = False)

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        """Minkowski sum of a VPolytope VP and another set or vector S.
        Defined as {a + s | a in VP, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Summand.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            VPolytope: Result of the Minkowski sum.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self + other
        elif not isinstance(other, VPolytope):
            other = VPolytope(**other.vpolytope(mode = mode))

        # add each combination
        V_sum = np.zeros((self.number_vertices()*other.number_vertices(), self.dimension))
        # todo: replace this by a faster method
        for i in range(self.number_vertices()):
            for j in range(other.number_vertices()):
                V_sum[i * other.number_vertices() + j] = self.V[i] + other.V[j]

        return VPolytope(V = V_sum, validate = False)
    
    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> VPolytope:
        """Minkowski difference between a VPolytope VP and another set or vector S.
        Defined as {s | s + S in VP}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Subtrahend.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Not implemented other than for single-point subtrahend.

        Returns:
            VPolytope: Result of the Minkowski difference.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other
        elif isinstance(other, VPolytope) and other.number_vertices() == 1:
            return self - np.reshape(other.V, (self.dimension, ))

        raise NotImplementedError
    
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
        """Projects a VPoltytope onto its own affine hull.
        For degenerate vpolytopes, the resulting vpolytope is of lower dimension, but non-degenerate.

        Returns:
            tuple: Projected VPolytope, projection matrix, center of new coordinate system in old coordinate system.
        """
        # compute basis of affine hull
        c = self.center()
        VP_shifted = self - c
        M_proj = VP_shifted.basis_affine_hull()
        # early exit if basis of affine hull is n-dimensional identity
        if np.array_equal(M_proj, np.eye(self.dimension)):
            return (self, M_proj, np.zeros(self.dimension))

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

    # representation by other set representation
    def represents(self, set_class: str, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope VP can also be equivalently represented using another ConvexSet class.

        Args:
            set_class (str): Name of another ConvexSet class or 'Point'.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Raises:
            NotImplementedError: Zonotope and Interval currently not supported.

        Returns:
            bool: Representation possible.
        """
        self._checkSetClass(set_class)

        if set_class == 'Point':
            if self.number_vertices() == 1:
                return True
            return np.allclose(self.V - self.V[0], 0., rtol = rtol, atol = atol)

        # 1D or single vertex always true
        if self.dimension == 1 or self.number_vertices() <= 1:
            return True
        
        if set_class in ['VPolytope', 'HPolyhedron']:
            return True
        
        if set_class == 'Interval':
            # todo: check if there is another way to do this...
            return self._represents_interval(rtol = rtol, atol = atol)
        
        if set_class == 'Zonotope':
            return self._represents_zonotope(rtol = rtol, atol = atol)
    
    # representation as an interval
    def _represents_interval(self, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope VP can also be equivalently represented as an Interval.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Representation possible.
        """
        # check whether all 2^n vertices of the interval outer approximation is contained in self
        interval_dict = self.interval(mode = 'outer')
        lower_bound, upper_bound = interval_dict['lb'], interval_dict['ub']
        # !note: functionality below is copied from interval/vertices...
        # reformat so that each dimension is a single np.ndarray (required for combinations below)
        bounds_per_dimension = np.vsplit(np.vstack((lower_bound, upper_bound)).transpose(), self.dimension)
        # remove second dimension for individual dimensions
        var = [x.flatten() for x in bounds_per_dimension]
        # flatten dimensions where lower bound equals the upper bound, write in tuple to unpack for itertools.product call
        t = tuple(x if x[0] != x[1] else np.array([x[0]]) for x in var)
        # enumerate all combinations
        all_combinations = product(*t)
        V = np.vstack([np.array(x) for x in all_combinations])
        # check for equality
        return self.__eq__(VPolytope(V = V, validate = False), rtol = rtol, atol = atol)

    def _represents_zonotope(self, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a VPolytope VP can also be equivalently represented as a Zonotope.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Representation possible.
        """
        # remove all vertices up to the given tolerance from the list of vertices
        self = self.compact()
        V_ = VPolytope(V = remove_duplicate_points(self.V, rtol = rtol, atol = atol), validate = False)

        # due to symmetry, zonotopes always have an even number of vertices
        m = V_.number_vertices()
        if m % 2 != 0:
            return False

        # idea: for each vertex, there must be another vertex across the center
        # 1. subtract the center from the list of vertices
        V = V_.V - V_.center()
        # 2. sort the list of vertices
        V = np.sort(V, axis = 0)
        # 3. compare top to bottom (must add up to 0)
        return np.allclose(V[0:int(m/2)] + V[-1:int(m/2)-1:-1], 0., rtol = rtol, atol = atol)

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
    def zonotope(self, *, mode: str = 'exact') -> dict:
        """Conversion to Zonotope. We use a box enclosure.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an Zonotope object.
        """
        self._checkMode(mode)

        if self.number_vertices() == 1:
            return {'c': self.V[0].flatten()}

        if mode == 'inner':
            if not self.represents('Interval'):
                raise NotImplementedError
            # else: use method below
        elif mode == 'exact':
            if not self.represents('Zonotope'):
                raise ExactEvaluationImpossibleError
            # don't know how to do exact conversion (method below is exact for 1D, though)
            if self.dimension != 1:
                raise NotImplementedError

        # convert to interval (outer approximation)
        interval_dict = self.interval(mode = 'outer')
        lower_bound, upper_bound = interval_dict['lb'], interval_dict['ub']

        # convert interval to zonotope (note: we cannot call Interval methods here)
        center = (upper_bound + lower_bound) / 2.
        generators = np.diag((upper_bound - lower_bound) / 2.)
        generators = generators[~np.all(generators == 0, axis=1), :]
        
        return {'c': center, 'G': generators}
