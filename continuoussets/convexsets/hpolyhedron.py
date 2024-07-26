from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linprog
from continuoussets.convexsets.convexset import ConvexSet
# from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError, ExactEvaluationImpossibleError

if __name__ == '__main__':
    print('This is the HPolyhedron class.')


class HPolyhedron(ConvexSet):
    # the operations in this class are taken from
    # [1] Wetzlinger et al. "Implementation of Polyhedral Operations in CORA 2024", ARCH'24.

    def __init__(self, *,
                 A: Union[np.ndarray, list, float, int] = None,
                 b: Union[np.ndarray, list, float, int] = None,
                 validate: bool = True):
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
    def __eq__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        """Set equality of an HPolyhedron HP with another set or vector S.
        Defined as forall i in HP: i in S and forall s in S: s in HP?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.

        Returns:
            bool: Set equality.
        """
        self._checkOtherOperand(other)

        # special check for comparison to vector
        if isinstance(other, np.ndarray):
            if not self.contains(other):
                return False
            raise NotImplementedError
        
        # convert everything to a HPolyhedron
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = 'exact'))

        # slow containment method
        return self.contains(other) and other.contains(self)
    
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
        self._checkOtherOperand(direction)

        raise NotImplementedError
    
    # check for boundedness
    def bounded(self) -> bool:
        """Checks if an HPolyhedron is bounded or not.

        Returns:
            bool: Boundedness.
        """
        # check if the support function value is finite in all directions of the nD simplex (eye(n), -1)
        value = self.support_function(-np.ones(self.dimension))
        if value == np.inf:
            return False
        elif value == -np.inf:  # empty -> bounded
            return True

        for i in range(self.dimension):
            basis_vector = np.zeros(self.dimension)
            basis_vector[i] = 1
            value = self.support_function(basis_vector)
            if value == np.inf:
                return False
            elif value == -np.inf:
                return True
            
        return True

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # convert other sets to Hpolyhedron
        if isinstance(other, np.ndarray):
            raise NotImplementedError
        
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

        # LP for Chebyshev center (raise error for unbounded sets)
        raise NotImplementedError
    
    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> HPolyhedron:

        # check for redundancy of each halfspace constraint
        raise NotImplementedError

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return np.all(np.matmul(self.A, other) <= self.b)
        
        for i in range(self.number_constraints()):
            # compute support function value of in-body along all normal vectors in A and compare to b
            (value, vector) = other.support_function(self.A[i])
            if value > self.b:
                return False
        
        return True

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'outer') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            raise NotImplementedError
        
        if mode == ['inner', 'exact']:
            raise NotImplementedError
        
        h = self.number_constraints()
        
        # 'outer': compute support function of self+other, take larger value, additional constraints from box
        A_new = np.vstack((self.A, np.eye(self.dimension), -np.eye(self.dimension)))
        b_new = np.zeros(h + 2*self.dimension)

        # for the first constraints, we already have the value computed for the HPolyhedron
        for i in range(h):
            # compute support function value of other set
            value_other = other.support_function(self.A[i])
            b_new[i] = self.b[i] if self.b[i] > value_other else value_other

        # for the remaining constraints, we also have to evaluate the support function for the HPolyhedron
        for i in range(2*self.dimension):
            value_polyhedron = self.support_function(self.A[h+i])
            value_other = other.support_function(self.A[h+i])
            b_new[i] = value_polyhedron if value_polyhedron < value_other else value_other

        return HPolyhedron(A = A_new, b = b_new)
    
    def empty(self) -> bool:
        """Checks if an HPolyhedron HP is empty.

        Returns:
            bool: Emptiness.
        """
        # objective function
        c = self.b

        # constraints
        A_eq = self.A
        b_eq = np.zeros(self.number_constraints())

        # solve linear program (bounds default (0, Inf) which is required here)
        res = linprog(c, A_eq = A_eq, b_eq = b_eq)

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
    
    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            return self.contains(other)

        # LPs for all other ConvexSet classes
        raise NotImplementedError

    # conversion to interval
    def interval(self, *, mode: str = 'exact') -> dict:
        self._checkMode(mode)

        if mode == 'inner':
            if not self.represents('Interval'):
                raise NotImplementedError
            # else: proceed with 'outer' conversion, which is exact
        if mode == 'exact':
            if not self.represents('Interval'):
                raise ExactEvaluationImpossibleError
        
        # support function in positive/negative axis-aligned directions
        # raise error if polyhedron is unbounded
        raise NotImplementedError
    
    # linear map
    def matmul(self, matrix: np.ndarray) -> HPolyhedron:
        self._checkMatrix(matrix)

        # case differentiation between square invertible matrices and projections
        raise NotImplementedError

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self + other
        
        # todo: for mode = 'outer', implement addition of support function evaluation
        
        if not isinstance(other, HPolyhedron):
            other = HPolyhedron(**other.hpolyhedron(mode = 'exact'))
        
        # 2. concatenation/lifting
        # 3. projection onto first n dimensions
        raise NotImplementedError
    
    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> HPolyhedron:
        """Minkowski difference between an HPolyhedron HP and another set or vector S.
        Defined as {s | s + S in HP}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            HPolyhedron: Result of the Minkowski difference.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other

        b_new = self.b
        for i in range(self.b):
            b_new[i] -= other.support_function(self.A[i])

        return HPolyhedron(A = self.A, b = b_new)
    
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

        # todo: implement fourier_motzkin helper function
        raise NotImplementedError

    # representation by other set representation
    def represents(self, *, set_class: str) -> bool:
        """Check if an HPolyhedron HP can also be equivalently represented using another ConvexSet class.

        Args:
            set_class (str): Name of another ConvexSet class.

        Returns:
            bool: Representation possible.
        """
        self._checkSetClass(set_class)

        if set_class == 'HPolyhedron':
            return True
        
        if set_class == 'VPolytope':
            return self.bounded()
        
        # Interval: minimal representation only axis-aligned constraints (all dims!)
        # Zonotope: raise NotImplementedError
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
            return (np.inf, [])
        elif res.status == 2:  # infeasible
            return (-np.inf, [])

        return (-res.fun, res.x)
    
    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of an HPolyhedron HP.

        Returns:
            np.ndarray: 2D array containing vertices as rows.
        """
        # obtain minimal representation
        # H = self.compact()

        # could be difficult... some built-in function?
        raise NotImplementedError

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
