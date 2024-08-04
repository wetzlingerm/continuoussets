from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog
from pypoman import compute_polytope_vertices  # project_polytope
from continuoussets.convexsets.convexset import ConvexSet
# from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError, ExactEvaluationImpossibleError, \
    EmptySetError, UnboundedSetError
from continuoussets.utils.auxiliary import halfspace_representation_from_vector, fourier_motzkin_elimination

if __name__ == '__main__':
    print('This is the HPolyhedron class.')


class HPolyhedron(ConvexSet):
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
            OtherFunctionError: If HPolyhedron + ConvexSet, call minkowski_sum instead.

        Returns:
            HPolyhedron: Result of the translation.
        """
        self._checkOtherOperand(other)
        
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return HPolyhedron(A = self.A, b = self.b + np.matmul(self.A, other), validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')
        
    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Set equality of an HPolyhedron HP with another set or vector S.
        Defined as forall i in HP: i in S and forall s in S: s in HP?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Set equality.
        """
        self._checkOtherOperand(other)

        # special check for comparison to vector
        if isinstance(other, np.ndarray):
            if not self.contains(other, rtol = rtol, atol = atol):
                return False
            A_other, b_other = halfspace_representation_from_vector(other)
            other = HPolyhedron(A = A_other, b = b_other)
            return other.contains(self, rtol = rtol, atol = atol)
        
        # convert everything to a HPolyhedron
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = 'exact'))

        # slow containment method
        a = self.contains(other, rtol = rtol, atol = atol)
        b = other.contains(self, rtol = rtol, atol = atol)
        return a and b
    
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
            return HPolyhedron(A = self.A, b = self.b - np.matmul(self.A, other), validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of an HPolyhedron HP in a given direction.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Raises:
            NotImplementedError: Currently not supported.

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        raise NotImplementedError
    
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

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        """Cartesian product of an HPolyhedron HP and another set or vector S.
        Defined as {[a^T s^T]^T | a in HP, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            HPolyhedron: Result of the Cartesian product.
        """
        self._checkMode(mode)

        # convert other sets to Hpolyhedron
        if isinstance(other, np.ndarray):
            A_other, b_other = halfspace_representation_from_vector(other)
            other = HPolyhedron(A = A_other, b = b_other)
        
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = 'exact'))

        # block-concatenation of constraint matrices, stack constraint offsets
        n1 = self.dimension
        n2 = other.dimension
        h1 = self.number_constraints()
        h2 = other.number_constraints()
        A_new = np.vstack((np.hstack((self.A, np.zeros((h1, n2)))),
                           np.hstack((np.zeros((h2, n1)), other.A))))
        b_new = np.hstack((self.b, other.b))

        return HPolyhedron(A = A_new, b = b_new, validate = False)

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

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks containment of a ConvexSet or vector (np.ndarray) S in an HPolyhedron HP.
        Defined as forall s in S: s in HP?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Containment.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            values = np.matmul(self.A, other)
            # check absolute tolerance... only then check relative tolerance
            if not np.all(values <= self.b + atol):
                min_value = np.min((np.abs(values), np.abs(self.b)), axis = 0)
                if np.any(min_value == 0.) or np.any(np.abs(values - self.b) / min_value > rtol):
                    return False
            return True
        
        for i in range(self.number_constraints()):
            # compute support function value of in-body along all normal vectors in A and compare to b
            value = other.support_function(self.A[i])[0]
            # check absolute tolerance... only then check relative tolerance
            if value > self.b[i] + atol:
                min_value = np.min((np.abs(value), np.abs(self.b[i])))
                if (min_value == 0.) or (np.abs(value - self.b[i]) / min_value > rtol):
                    return False
        
        return True

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        """Convex hull of an HPolyhedron HP and another set or vector S.
        Defined as {lambda*h + (1-lambda)*s | h in HP, s in S, lambda in [0,1]}

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Convex hull with vector not supported.
            NotImplementedError: mode in ['inner', 'exact'] not supported.

        Returns:
            HPolyhedron: Result of the convex hull.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            A_other, b_other = halfspace_representation_from_vector(other)
            other = HPolyhedron(A = A_other, b = b_other)
        
        if mode in ['inner', 'exact']:
            raise NotImplementedError
        
        h = self.number_constraints()
        
        # 'outer': compute support function of self+other, take larger value, additional constraints from box
        A_new = np.vstack((self.A, np.eye(self.dimension), -np.eye(self.dimension)))
        b_new = np.zeros(h + 2*self.dimension)

        # for the first constraints, we already have the value computed for the HPolyhedron
        for i in range(h):
            # compute support function value of other set
            value_other = other.support_function(self.A[i])[0]
            b_new[i] = self.b[i] if self.b[i] > value_other else value_other

        # for the remaining constraints, we also have to evaluate the support function for the HPolyhedron
        for i in range(2*self.dimension):
            value_polyhedron = self.support_function(A_new[h+i])[0]
            value_other = other.support_function(A_new[h+i])[0]
            b_new[h+i] = value_polyhedron if value_polyhedron > value_other else value_other

        return HPolyhedron(A = A_new, b = b_new, validate = False)
    
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
            # todo: unbounded sets may be degenerate
            raise NotImplementedError

        return np.any(np.isclose(self.b - np.matmul(self.A, c), 0., rtol = rtol, atol = atol))
    
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

        if res.status == 2:
            return False
        elif res.status == 3:
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
    
    # intersection
    def intersection(self, other: Union[ConvexSet, np.ndarray], mode: str = 'exact') -> HPolyhedron:
        """Computation of the intersection of an HPolyhedron HP and another set or vector S.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            HPolyhedron: Result of the intersection.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            A_other, b_other = halfspace_representation_from_vector(other)
            return self.intersection(HPolyhedron(A = A_other, b = b_other))
            
        # convert all other sets to HPolyhedron
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = mode))

        return HPolyhedron(A = np.vstack((self.A, other.A)),
                           b = np.hstack((self.b, other.b)),
                           validate = False)
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if an HPolyhedron intersects another set of vector S.
        Defined as exists s in HP: s in S?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)
        elif isinstance(other, HPolyhedron):
            return self._intersects_hpolyhedron(other, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'Interval':
            return self._intersects_interval(other, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'Zonotope':
            return self._intersects_zonotope(other, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'VPolytope':
            return self._intersects_vpolytope(other, rtol = rtol, atol = atol)
    
    # intersection check with hpolyhedron
    def _intersects_hpolyhedron(self, other: HPolyhedron, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Intersection of an HPolyhedron HP with another HPolyhedron S.

        Args:
            other (HPolyhedron): HPolyhedron.

        Returns:
            bool: Result of the intersection check.
        """
        # compute explicit intersection and check whether it is empty
        return not self.intersection(other).empty()

    # intersection check with interval
    def _intersects_interval(self, other, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Intersection of an HPolyhedron HP with an Interval I.

        Args:
            other (Interval): Interval.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # linear program: min 0  s.t.  Ax <= b, lb <= x <= ub
        res = linprog(np.zeros(self.dimension),
                      A_ub = np.vstack((self.A, np.eye(self.dimension), -np.eye(self.dimension))),
                      b_ub = np.hstack((self.b, other.ub, -other.lb)),
                      bounds = (None, None))
        return res.success
    
    # intersection check with zonotope
    def _intersects_zonotope(self, other, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Intersection of an HPolyhedron HP with a Zonotope Z.

        Args:
            other (Zonotope): Zonotope.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # linear program: min 0  s.t.  Ax <= b, c + Gbeta == x, ||beta||_oo <= 1
        n, m = self.dimension, other.number_generators()

        c = np.zeros(n + m)
        A_ub = np.vstack((np.hstack((self.A, np.zeros((self.number_constraints(), m)))),
                          np.hstack((np.zeros((2*m, n)),
                                     np.vstack((np.eye(m), -np.eye(m)))))))
        b_ub = np.hstack((self.b, np.ones(2*m)))
        A_eq = np.hstack((-np.eye(n), other.G.T))
        b_eq = -other.c

        res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = (None, None))
        return res.success
    
    def _intersects_vpolytope(self, other, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Intersection of an HPolyhedron HP with a VPolytope VP.

        Args:
            other (VPolytope): VPolytope.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Result of the intersection check.
        """
        # linear program: min 0  s.t.  Ax <= b, Vbeta == x, sum beta = 1, beta >= 0
        n, h, m = self.dimension, self.number_constraints(), other.number_vertices()

        c = np.zeros(n + m)
        A_ub = np.vstack((np.hstack((self.A, np.zeros((h, m)))),
                          np.hstack((np.zeros((m, n)), -np.eye(m)))))
        b_ub = np.hstack((self.b, np.zeros(m)))
        A_eq = np.vstack((np.hstack((-np.eye(n), other.V.T)),
                          np.hstack((np.zeros(n), np.ones(m)))))
        b_eq = np.hstack((np.zeros(n), 1.))

        res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = (None, None))
        return res.success

    # conversion to interval
    def interval(self, *, mode: str = 'exact') -> dict:
        """Conversion of an HPolyhedron HP to an Interval I.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: mode = 'inner' not supported unless HP represents an interval.
            ExactEvaluationImpossibleError: mode = 'exact' not supported unless HP represents an interval.
            UnboundedSetError: HPolyhedron is unbounded.
            EmptySetError: HPolyhedron is empty.

        Returns:
            dict: Keyword arguments for instantiation of an Interval object.
        """
        self._checkMode(mode)

        if mode == 'inner':
            if not self.represents(set_class = 'Interval'):
                raise NotImplementedError
            # else: proceed with 'outer' conversion, which is exact
        if mode == 'exact':
            if not self.represents(set_class = 'Interval'):
                raise ExactEvaluationImpossibleError
        
        # loop over all 2n -+ basis vectors and use support function value
        n = self.dimension
        lower_bound = np.zeros(n)
        upper_bound = np.zeros(n)

        basis_vector = np.zeros(n)
        for i in range(n):
            for s in [-1., 1.]:
                basis_vector[i] = s
                value = self.support_function(basis_vector)[0]
                basis_vector[i] = 0.
                if value == np.inf:
                    raise UnboundedSetError
                elif value == -np.inf:
                    raise EmptySetError
                elif s == -1.:
                    lower_bound[i] = -value
                else:  # s == 1.
                    upper_bound[i] = value
        
        return {'lb': lower_bound, 'ub': upper_bound}
    
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
                      np.block([[D_inv, np.zeros((r,n-r))], [np.zeros((n-r,r)), np.eye(n-r)]]))
        # 4. project onto first r dimensions
        P = HPolyhedron(A = A_new, b = self.b.copy())
        P = P.project(axis = tuple(np.arange(r)))
        # 5. multiply with orthogonal matrix
        P = P.matmul(U)
        return P

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        """Minkowski sum of an HPolyhedron HP and another set or vector S.
        Defined as {a + s | a in HP, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Summand.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            HPolyhedron: Result of the Minkowski sum.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self + other
        
        n = self.dimension
        h = self.number_constraints()

        if mode == 'outer':
            # addition of support function evaluation
            A_new = np.vstack((self.A, np.eye(n), -np.eye(n)))
            b_new = np.hstack((self.b, np.zeros(2*n)))
        
            # first h constraints: only compute support function of other
            for i in range(h):
                b_new[i] += other.support_function(A_new[i])[0]

            # remaining 2n constraints: also compute support function of self
            for i in range(2*n):
                b_new[h+i] = self.support_function(A_new[h+i])[0] + other.support_function(A_new[h+i])[0]

            return HPolyhedron(A = A_new, b = b_new, validate = False)
        
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = 'exact'))
        
        # 2. concatenation/lifting
        HP_lifted = self.cartesian_product(other)

        # 3. projection onto first n dimensions
        M = np.hstack((np.eye(n), np.eye(n)))
        return HP_lifted.matmul(M)
    
    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        """Minkowski difference between an HPolyhedron HP and another set or vector S.
        Defined as {s | s + S in HP}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            HPolyhedron: Result of the Minkowski difference.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other

        b_new = self.b.copy()
        for i in range(self.number_constraints()):
            b_new[i] -= other.support_function(self.A[i])[0]

        return HPolyhedron(A = self.A.copy(), b = b_new)
    
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

    # representation by other set representation
    def represents(self, set_class: str, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if an HPolyhedron HP can also be equivalently represented using another ConvexSet class.

        Args:
            set_class (str): Name of another ConvexSet class or 'Point'.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Representation possible.
        """
        self._checkSetClass(set_class)

        if set_class == 'Point':
            try:
                c = self.center()
            except (UnboundedSetError, EmptySetError):
                return False
            # center must fulfill all inequalities with equality
            return np.allclose(np.matmul(self.A, c), self.b, rtol = rtol, atol = atol)

        if set_class == 'HPolyhedron':
            return True

        # all bounded 1D sets can be represented by every set representation exactly
        if self.dimension == 1:
            return self.bounded()
        
        # vpolytopes can only represent bounded polyhedra
        if set_class == 'VPolytope':
            return self.bounded()
        
        if set_class == 'Interval':
            return self._represents_interval(rtol = rtol, atol = atol)
        
        # todo Zonotope...
        raise NotImplementedError
    
    def _represents_interval(self, *, rtol: float = 1e-5, atol: float = 1e-8):
        """Check if an HPolyhedron HP can also be equivalently represented by an Interval.

        Args:
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolut tolerance. Defaults to 1e-8.

        Returns:
            _type_: Representation possible.
        """
        # todo: what to do with empty?

        # must have at least 2n constraints
        if self.number_constraints() < 2*self.dimension:
            return False
        
        # keep indices for redundancy and for which dimensions are bounded
        index_keep_for_i = np.full((self.number_constraints(),), True)
        bounded_dimensions_plus = np.full((self.dimension,), False)
        bounded_dimensions_minus = np.full((self.dimension,), False)

        # loop over constraints, if not axis-aligned -> must be redundant
        for i in range(self.number_constraints()):
            # axis-aligned constraint may only have a single non-zero entry
            non_zero_entry = np.invert(np.isclose(self.A[i], 0., rtol = rtol, atol = atol))

            if np.sum(non_zero_entry) > 1:
                # check if constraint is redundant
                index_keep_for_i[i] = False
                polyhedron_i = HPolyhedron(A = self.A[index_keep_for_i], b = self.b[index_keep_for_i])
                value = polyhedron_i.support_function(self.A[i])[0]
                if value > self.b[i] + atol:
                    return False
                # note: we do not have to reset index_keep_for_i, as thee ith constraint is redundant

            else:
                # append this dimension to the respective list of bounded dimensions
                if self.A[i][non_zero_entry] > 0:
                    bounded_dimensions_plus = np.logical_or(bounded_dimensions_plus, non_zero_entry)
                else:
                    bounded_dimensions_minus = np.logical_or(bounded_dimensions_minus, non_zero_entry)

        # currently, all dimensions must be bounded
        return np.all(bounded_dimensions_plus) and np.all(bounded_dimensions_minus)
    
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
        # H = self.compact()

        if self.empty():
            raise EmptySetError
        elif self.degenerate():
            # todo: implement...
            raise NotImplementedError

        # non-degenerate case
        V = compute_polytope_vertices(self.A, self.b)
        return np.reshape(V, (len(V), self.dimension))

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
    
    # conversion to vpolytope
    def vpolytope(self, *, mode: str = 'exact') -> dict:
        """Conversion to a VPolytope VP.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of a VPolytope object.
        """
        self._checkMode(mode)

        return {'V': self.vertices()}

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'exact') -> dict:
        """Conversion to a Zonotope Z.

        Args:
            mode (str, optional): Approximation of conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: keyword arguments for instantiation of a Zonotope object
        """
        self._checkMode(mode)

        # call conversion to interval and convert interval to zonotope
        raise NotImplementedError
        # return {'c': ..., 'G': ...}
