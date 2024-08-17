from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog
from pypoman import compute_polytope_vertices
from continuoussets.convexsets.interface_convexset import IConvexSet
# from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError, EmptySetError, UnboundedSetError
from continuoussets.utils.auxiliary import fourier_motzkin_elimination, active_inequality

if __name__ == '__main__':
    print('This is the HPolyhedron class.')


class HPolyhedron(IConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *,
                 A: Union[np.ndarray, list, float, int] = None,
                 b: Union[np.ndarray, list, float, int] = None,
                 validate: bool = True):
        """Instantiates an HPolyhedron object HP = {x | Ax <= b}.

        Args:
            A (Union[np.ndarray, list, float, int], optional): Constraint matrix. Defaults to None.
            b (Union[np.ndarray, list, float, int], optional): Constraint offset. Defaults to None.
            validate (bool, optional): Input argument check. Defaults to True.

        Raises:
            ValueError: Two inputs are required.
            TypeError: Constraint matrix must be int, float, list or np.ndarray.
            TypeError: Constraint offset must be int, float, list or np.ndarray.
            ValueError: Constraint offset needs to be a 1D array.
            ValueError: Constraint matrix must be a 1D or 2D array.
            ValueError: Columns in constraint matrix must match size of constraint offset.

        Returns:
            HPolyhedron: Polyhedron.
        """
        # constraint matrix and constraint offset have to be given
        if self.validate and validate:
            if A is None or b is None:
                raise ValueError('HPolyhedron:__init__',
                                 'The constructor requires two input arguments')
            if (not isinstance(A, int) and not isinstance(A, float)
                    and not isinstance(A, list) and not isinstance(A, np.ndarray)):
                raise TypeError('HPolyhedron:__init__',
                                'Constraint matrix must be int, float, list or np.ndarray')
            elif (not isinstance(b, int) and not isinstance(b, float)
                    and not isinstance(b, list) and not isinstance(b, np.ndarray)):
                raise TypeError('HPolyhedron:__init__',
                                'Constraint offset must be int, float, list or np.ndarray')
            elif isinstance(b, np.ndarray) and b.ndim > 1:
                raise ValueError('HPolyhedron:__init__',
                                 'Constraint offset needs to be a 1D array.')

        # convert to numpy if possible
        if not isinstance(A, np.ndarray):
            if isinstance(A, int) or isinstance(A, float):
                A = np.reshape(np.array([float(A)]), (1, 1))
            elif isinstance(A, list):
                A = np.array(A, dtype = float)
        if not isinstance(b, np.ndarray):
            b = np.array(b)

        # expand to 2D array
        if A.ndim == 1:
            A = np.reshape(A, (1, A.size))

        # post-check: no higher than 2D
        if self.validate and validate:
            if A.ndim > 2:
                raise ValueError('HPolyhedron:__init__',
                                 'Constraint matrix must be a 1D or 2D array.')
            elif b.size != A.shape[0]:
                raise ValueError('HPolyhedron:__init__',
                                 'Number of constraints differs between constraint matrix and constraint offset.')

        self.dimension = A.shape[1]
        self.A = A.copy()
        self.b = b.copy()

    # deep copy
    def copy(self) -> HPolyhedron:
        """Returns a deep copy of an HPolyhedron.

        Returns:
            HPolyhedron: Copied HPolyhedron.
        """
        return HPolyhedron(A = self.A.copy(), b = self.b.copy(), validate = False)

    # display
    def __repr__(self):
        """Representation on the command window.

        Returns:
            str: Description of the HPolyhedron object.
        """
        newline = '\n'
        return f'{newline}dimension: {self.dimension}{newline}A: {self.A}{newline}b: {self.b}{newline}'

    # translation by vector
    def __add__(self, other: np.ndarray) -> HPolyhedron:
        """Translation of a HPolyhedron by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If HPolyhedron + IConvexSet, call minkowski_sum instead.

        Returns:
            HPolyhedron: Result of the translation.
        """
        self._checkOtherOperand(other)
        
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return HPolyhedron(A = self.A, b = self.b + np.matmul(self.A, other), validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
    
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
            OtherFunctionError: If HPolyhedron - IConvexSet, call minkowski_difference instead.

        Returns:
            HPolyhedron: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return HPolyhedron(A = self.A, b = self.b - np.matmul(self.A, other), validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # basis of the affine hull (for degenerate sets)
    def basis_affine_hull(self) -> tuple:
        """Computes a basis of the affine hull of an HPolyhedron HP.

        Returns:
            tuple: Matrix with basis vectors, number of required basis vectors.
        """
        # check if degenerate at all via Chebyshev center
        c = self.center()
        if not active_inequality(self.A, self.b, c):
            return (np.eye(self.dimension), self.dimension)

        # ensure polytope contains the origin
        HP_shift = self
        if not HP_shift._contains_origin():
            HP_shift = self - c

        # threshold for norm of next basis vector
        epsilon = 1e-5
        basis = np.zeros((self.dimension, 0))

        # loop over all dimensions
        for r in range(self.dimension):
            # compute next basis vector
            x_iter = HP_shift._basis_affine_hull_helper(basis)
            if x_iter is None or np.linalg.norm(x_iter) < epsilon:
                break
            basis = np.hstack((basis, np.reshape(x_iter / np.linalg.norm(x_iter, ord=2), (self.dimension, 1))))

        # fill in remaining dimensions via QR decomposition
        Q, _ = np.linalg.qr(np.hstack((basis, np.eye(self.dimension)[:, :r])))
        return (Q, r)

    # helper linear program for basis of affine hull
    def _basis_affine_hull_helper(self, basis: np.ndarray) -> np.ndarray:
        """Helper function for basis of affine hull.
        This function is only to be called in basis_affine_hull.

        Args:
            basis (np.ndarray): Current basis of affine hull.

        Returns:
            np.ndarray: Potential additional basis vector of affine hull.
        """
        # LP to find next basis vector
        
        # init maximum value
        max_value = 0.
        max_vector = None

        # loop over all n dimensions
        direction = np.zeros(self.dimension)
        for i in range(self.dimension):
            # loop over plus and minus
            for s in [1., -1.]:
                direction[i] = s
                (value, vector) = self._basis_affine_hull_helper_axis(basis, direction)
                direction[i] = 0
                if -value > max_value and not np.isclose(-value, 0., atol = 1e-10):
                    max_value = -value
                    max_vector = vector

        return max_vector
    
    def _basis_affine_hull_helper_axis(self, basis: np.ndarray, direction: np.ndarray) -> tuple:
        """Auxiliary linear program to find a potential additional basis vector of the affine hull.
        This function is only to be called in _basis_affine_hull_helper.

        Args:
            basis (np.ndarray): Current basis of the affine hull.
            direction (np.ndarray): Axis-aligned direction.

        Returns:
            tuple: Value and basis vector (potentially None).
        """
        # LP for given direction e
        # max_{x in R^n, w in R^r} np.dot(e, x)
        # s.t.  self.A (x + Bw) <= self.b  <=>  self.A * x + self.A*B * w <= self.b
        #       i in {1,...,r}: np.dot(B[i], x) == 0

        # dimension of current basis
        r = basis.shape[1]

        # objective function
        c = np.hstack((-direction, np.zeros(r)))

        # constraints
        A_ub = np.hstack((self.A, np.matmul(self.A, basis)))
        b_ub = self.b
        A_eq = np.hstack((basis.T, np.zeros((r, r))))
        b_eq = np.zeros(r)

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        return (res.fun, res.x[:self.dimension])

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of an HPolyhedron HP in a given direction.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Raises:
            NotImplementedError: HPolyhedron must contain the origin.
            NotImplementedError: HPolyhedron must be non-degenerate.
            UnboundedSetError: HPolyhedron is unbounded in the given direction.

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        if self.degenerate():
            raise NotImplementedError
        elif not self._contains_origin():
            raise NotImplementedError

        # LP formulation for boundary point computation
        # min_{x,l} -l
        # s.t.      A x <= b
        #           - x + l*dir = 0

        # read out dimension
        n, h = self.dimension, self.number_constraints()

        # objective function
        c = np.hstack((np.zeros(n), -1.))

        # constraints
        A_ub = np.hstack((self.A, np.zeros((h, 1))))
        b_ub = self.b
        A_eq = np.hstack((-np.eye(n), np.reshape(direction, (n, 1))))
        b_eq = np.zeros(n)

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        if res.status == 3:
            raise UnboundedSetError
        
        return -res.fun * direction
    
    # check for boundedness
    def bounded(self) -> bool:
        """Checks if an HPolyhedron is bounded.

        Returns:
            bool: Boundedness.
        """
        # check if the support function value is finite in all directions of the nD simplex (eye(n), -1)
        value = self.support_function(-np.ones(self.dimension))[0]
        if value == np.inf:  # unbounded
            return False
        elif value == -np.inf:  # empty -> bounded
            return True

        for i in range(self.dimension):
            basis_vector = np.zeros(self.dimension)
            basis_vector[i] = 1
            value = self.support_function(basis_vector)[0]
            if value == np.inf:  # unbounded
                return False
            
        return True

    # center
    def center(self) -> np.ndarray:
        """Computation of the Chebyshev center of an HPolyhedron HP via linear programming.
        Defined as the center with the ball of largest radius contained in HP.

        Raises:
            EmptySetError: Set is empty.
            UnboundedSetError: Set is unbounded. LP could not converge.

        Returns:
            np.ndarray: Chebyshev center.
        """
        # objective function
        c = np.hstack((-1, np.zeros(self.dimension)))

        # inequality constraints
        A_ub = np.vstack((np.hstack((-1, np.zeros(self.dimension))),
                          np.hstack((np.reshape(np.linalg.norm(self.A, axis=1, ord=2), (self.number_constraints(), 1)), self.A))))
        b_ub = np.hstack((0, self.b))

        # solve linear program
        res = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = (None, None))

        # check empty and unbounded cases
        if res.status == 2:  # infeasible -> empty
            raise EmptySetError
        elif res.status == 3:  # unbounded
            raise UnboundedSetError
        
        return res.x[1:]
    
    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> HPolyhedron:
        """Minimal representation of an HPolyhedron HP.
        Removes all redundant inequality constraints from the constraint matrix and constraint vector.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-12.

        Returns:
            HPolyhedron: HPolyhedron in minimal representation.
        """
        index_irredundant = np.full((self.number_constraints(),), False)
        index_keep_for_i = np.full((self.number_constraints(),), True)

        for i in range(self.number_constraints()):
            if not index_irredundant[i]:
                # evaluate support function in direction of ith normal vector
                # for polyhedron without ith constraint
                index_keep_for_i[i] = False
                polyhedron_i = HPolyhedron(A = self.A[index_keep_for_i], b = self.b[index_keep_for_i])
                (value, vector) = polyhedron_i.support_function(self.A[i])

                # compare to value of support function
                if self.b[i] < value:
                    # ith constraint is irredundant
                    index_irredundant[i] = True
                    index_keep_for_i[i] = True
                else:
                    # support vector x is a vertex of the (minimal) polyhedron
                    # -> constraints that fulfill Ax = b (with equality!) are irredundant as well
                    if vector is not None:  # avoid empty/unbounded cases
                        index_irredundant = np.logical_or(index_irredundant,
                                                          np.isclose(np.matmul(self.A, vector), self.b))
        
        return HPolyhedron(A = self.A[index_irredundant], b = self.b[index_irredundant], validate = False)
    
    # specific containment check
    def _contains_origin(self, *, atol: float = 1e-12):
        """Checks if the origin is contained in an HPolyhedron.

        Args:
            atol (float, optional): Absolute tolerance. Defaults to 1e-12.

        Returns:
            bool: Origin contained in HPolyhedron.
        """
        return np.all(self.b + atol > 0)

    # degeneracy
    def degenerate(self, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if an HPolyhedron is degenerate.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Degeneracy.
        """
        # compute Chebyshev center and check if it fulfills any inequality with equality
        try:
            c = self.center()
        except EmptySetError:
            # we consider empty sets to be degenerate
            return True
        except UnboundedSetError:
            # Chebyshev center has to be computable for degenerate case, because the radius
            # of the associated ball cannot grow over 0 -> otherwise non-degenerate
            return False

        return active_inequality(self.A, self.b, c, rtol = rtol, atol = atol)
    
    # emptiness
    def empty(self) -> bool:
        """Checks if an HPolyhedron HP is empty.

        Returns:
            bool: Emptiness.
        """
        # objective function
        c = self.b

        # constraints
        A_eq = self.A.T
        b_eq = np.zeros(self.dimension)

        # solve linear program (bounds default (0, Inf) which is required here)
        res = linprog(c, A_eq = A_eq, b_eq = b_eq)

        # cannot be infeasible since res.x = 0 fulfills any system of equalities A_eq * res.x = 0
        if res.status == 3:
            return True
        return res.fun < 0
    
    # conversion to zonotope
    def hpolyhedron(self, *, mode: str = 'exact') -> dict:
        """Overloaded conversion to HPolyhedron.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an HPolyhedron object.
        """
        self._checkMode(mode)

        return {'A': self.A, 'b': self.b}

    # linear map
    def matmul(self, matrix: np.ndarray) -> HPolyhedron:
        """Linear map of an HPolyhedron HP by a matrix (np.ndarray).
        Defined as {M s | s in HP}.

        Args:
            matrix (np.ndarray): Matrix for left-multiplication.

        Returns:
            Interval: Result of the matrix multiplication.
        """
        self._checkMatrix(matrix)

        m, n = matrix.shape
        if m == n and np.linalg.matrix_rank(matrix) == n:
            # simple formula for square and invertible matrices
            return HPolyhedron(A = np.matmul(self.A, np.linalg.inv(matrix)), b = self.b, validate = False)
        elif m > n:
            # projections to higher-dimensional space not supported
            raise NotImplementedError
        
        # general formula including projection:
        # 1. compute SVD and number of non-zero singular values
        U, S, V = np.linalg.svd(matrix)
        r = S.size - np.count_nonzero(np.isclose(S, 0., atol = 1e-8))
        # 2. compute diagonal matrix with 1/s
        D_inv = np.diag(1. / S)
        # 3. init polytope before projection
        A_new = np.matmul(np.matmul(self.A, V.T),
                          np.block([[D_inv, np.zeros((r, n-r))], [np.zeros((n-r, r)), np.eye(n-r)]]))
        # 4. project onto first r dimensions
        P = HPolyhedron(A = A_new, b = self.b.copy())
        P = P.project(axis = tuple(np.arange(r)))
        # 5. multiply with orthogonal matrix
        P = P.matmul(U)
        return P
    
    # number of constraints
    def number_constraints(self) -> int:
        """Returns the number of constraints in the constraint matrix.

        Returns:
            int: Number of constraints.
        """
        return self.b.size
    
    # projection onto subspace
    def project(self, *, axis: tuple) -> HPolyhedron:
        """Projection of an HPolytope HP onto a subspace.

        Args:
            axis (tuple): Subspace for projection.

        Returns:
            HPolyhedron: Projected HPolyhedron.
        """
        self._checkSubspace(axis)

        A_new, b_new = self.A.copy(), self.b.copy()
        count = 0
        for i in range(self.dimension):
            if i not in axis:
                A_new, b_new = fourier_motzkin_elimination(A_new, b_new, i-count)
                count += 1

        return HPolyhedron(A = A_new, b = b_new, validate = False)
    
    # projection onto its own affine hull
    def project_affine_hull(self) -> tuple:
        """Projects an HPolyhedron onto its own affine hull.
        For a degenerate HPolyhedron, the resulting HPolyhedron is of lower dimension, but non-degenerate.

        Returns:
            tuple: Projected HPolyhedron, projection matrix, center of new coordinate system in old coordinate system.
        """
        # compute basis of affine hull
        n, c = self.dimension, self.center()
        HP_shifted = self - c
        M_proj, r = HP_shifted.basis_affine_hull()
        # early exit if basis of affine hull is n-dimensional identity
        if r == n:
            return (self, np.eye(n), np.zeros(n))

        # map polyhedron onto lower dimensional space
        HP_proj = HP_shifted.matmul(M_proj.T)
        # remove dimensions from r to n
        A_new = HP_proj.A[:, :r]
        b_new = HP_proj.b
        # remove potential all-zero constraints
        non_flat = np.invert(np.all(np.isclose(A_new, 0.), axis = 1))
        A_new = A_new[non_flat, :]
        b_new = b_new[non_flat]
        # init resulting polyhedron
        HP_proj = HPolyhedron(A = A_new, b = b_new)

        return (HP_proj, M_proj, c)

    # reduction of set representation size
    def reduce(self, *, order: int) -> HPolyhedron:
        """Reduction of the set representation size of an HPolyhedron HP.

        Args:
            order (int): Reduced number of constraints.

        Raises:
            NotImplementedError: Currently not supported.

        Returns:
            HPolyhedron: HPolyhedron with reduced set representation size.
        """
        raise NotImplementedError

    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Support function evaluation of a HPolyhedron HP in a direction d.
        Value defined as max_{s in HP} d^T * s.
        Vector defined as arg max_{s in HP} d^T * s.
        Unbounded cases yield a value of np.inf, empty cases a value of -np.inf.
        Both cases do not return a meaningful support vector.

        Args:
            direction (np.ndarray): Direction along which to evaluate the support function.

        Returns:
            tuple[float, np.ndarray]: Support value and support vector.
        """
        self._checkOtherOperand(direction)

        # LP for value and vector
        res = linprog(-direction, A_ub = self.A, b_ub = self.b, bounds = (None, None))
        if res.status == 3:  # unbounded
            return (np.inf, None)
        elif res.status == 2:  # infeasible
            return (-np.inf, None)

        return (-res.fun, res.x)
    
    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of an HPolyhedron HP.

        Returns:
            np.ndarray: 2D array containing vertices as rows.
        """
        # obtain minimal representation
        HP = self.compact()

        if self.empty():
            raise EmptySetError
        
        n_orig = self.dimension
        if HP.degenerate():
            # map set into the basis of its affine hull where it is full-dimensional
            (HP, M_proj, c) = HP.project_affine_hull()

        # non-degenerate case
        n = HP.dimension
        try:
            V = compute_polytope_vertices(HP.A, HP.b)
        except (ValueError):
            # ValueError occurs in unbounded cases (in pypoman/duality.py)
            raise UnboundedSetError
        # format vertices correctly
        V = np.reshape(V, (len(V), n))

        # map back to original higher-dimensional space
        if n_orig > n:
            # expand vertex matrix by zeros for all projected dimensions
            V = np.hstack((V, np.zeros((V.shape[0], n_orig-n))))
            # map by inversion (M_proj^-1 = M_proj.T) of previous mapping, incorporate effect of center
            V = np.matmul(M_proj, V.T).T + c

        return V

    # volume
    def volume(self) -> float:
        """Computation of the volume of an HPolyhedron HP.
        Defined as 0 if HP is empty or degenerate, defined as np.inf if HP is unbounded.

        Returns:
            float: Volume.
        """
        if not self.bounded():
            return np.inf
        elif self.empty() or self.degenerate():
            return 0
        
        raise NotImplementedError
