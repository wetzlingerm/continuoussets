from __future__ import annotations

from itertools import combinations
from typing import Union

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError, ExactEvaluationImpossibleError

if __name__ == '__main__':
    print('This is the Zonotope class.')


class Zonotope(ConvexSet):

    # constructor
    def __init__(self, *, c: Union[np.ndarray, list, float, int] = None,
                 G: Union[np.ndarray, list, float, int] = None, validate: bool = True):
        """Instantiates a Zonotope object Z = {c + sum G_i beta_i | -1 <= beta_i <= 1}.

        Args:
            c (Union[np.ndarray, list, float, int], optional): Center of the zonotope. Defaults to None.
            G (Union[np.ndarray, list, float, int], optional): 2D generator matrix of the zonotope, each generator is a row.
            Defaults to None.
            validate (bool, optional): Validation of input arguments. Defaults to True.

        Raises:
            ValueError: No input arguments provided.
            ValueError: No center defined.
            ValueError: Center must be a scalar or vector.
            ValueError: Dimensions of center and generator matrix must match.
        """

        # check center
        if self.validate and validate:
            # at least center has to be provided
            if c is None:
                raise ValueError('Zonotope:__init__',
                                 'Center has to be defined.')
            elif (not isinstance(c, int) and not isinstance(c, float)
                    and not isinstance(c, list) and not isinstance(c, np.ndarray)):
                raise TypeError('Zonotope:__init__',
                                'Center must be int, float, list or np.ndarray')
            elif isinstance(c, np.ndarray) and c.ndim > 1:
                raise ValueError('Zonotope:__init__',
                                 'Center needs to be a 1D array.')

        # convert center to np.ndarray
        if not isinstance(c, np.ndarray):
            if isinstance(c, int) or isinstance(c, float):
                c = np.array([float(c)])
            elif isinstance(c, list):
                c = np.array(c, dtype = float)

        # init 0-width matrix (for concatenation in methods)
        if G is None:
            G = np.zeros((0, c.size))

        # pre-check generator matrix
        if self.validate and validate:
            if (not isinstance(G, int) and not isinstance(G, float)
                    and not isinstance(G, list) and not isinstance(G, np.ndarray)):
                raise TypeError('Zonotope:__init__',
                                'Generator matrix must be None, int, float, list or np.ndarray')

        # convert generator(s) to np.ndarray
        if not isinstance(G, np.ndarray):
            if isinstance(G, int) or isinstance(G, float):
                G = np.reshape(np.array([float(G)]), (1, 1))
            elif isinstance(G, list):
                G = np.array(G, dtype = float)

        # expand generator matrix to 2D if only single generator
        if G.ndim == 1:
            G = np.reshape(G, (1, G.size))

        # post-check generator matrix (ensured to be np.ndarray now)
        if self.validate and validate:
            if c.size != G.shape[1]:
                raise ValueError('Zonotope:__init__',
                                 'Center and generator matrix need to have the same dimension.')

        self.dimension = c.size
        self.c = c.copy()
        self.G = G.copy()

    # display
    def __repr__(self) -> str:
        """Representation on the command window.

        Returns:
            str: Description of the Zonotope object.
        """
        newline = '\n'
        return f'dimension: {self.dimension}{newline}center:{newline} {self.c}{newline}generator matrix:{newline} {self.G}'

    # enable correct handling of right-operations with numpy on left side
    def __array_ufunc__(self, ufunc, method: str, *args, **kwargs) -> Zonotope:
        """To enable the correct handling of operations with a numpy object on the left side
        with a Zonotope object Z, i.e., np.array([1., 2.]) - Z.

        Args:
            ufunc (np.ufunc): Called ufunc object.
            method (str): Indication which Ufunc method was called. Here: '__call__'.

        Raises:
            NotImplementedError: Only right-operations 'add' and 'subtract' are supported.

        Returns:
            Zonotope: Result of the respective arithmetic operation.
        """
        if ufunc.__name__ == 'add':
            # should be __radd__ -> re-order and call __add__
            return args[1] + args[0]
        elif ufunc.__name__ == 'subtract':
            # should be __rsub__ -> convert to zonotope and call __add__
            return args[0] + (-args[1])
        else:
            raise NotImplementedError

    # translation by vector
    def __add__(self, other: np.ndarray) -> Zonotope:
        """Translation of a Zonotope by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If other is a ConvexSet, call minkowski_sum instead.

        Returns:
            Zonotope: Result of the translation.
        """
        # TODO support int, float, list
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return Zonotope(c = self.c + other, G = self.G, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')

    # set equality
    def __eq__(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
        """Set equality of a Zonotope Z with another set or vector S.
        Defined as forall Z in Z: i in S and forall s in S: s in Z?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-10.
            atol (float, optional): Absolute tolerance. Defaults to 1e-12.

        Returns:
            bool: Set equality.
        """
        self._checkOtherOperand(other, check_dimension = False)

        if ((isinstance(other, ConvexSet) and self.dimension != other.dimension)
                or (isinstance(other, np.ndarray) and self.dimension != other.shape[0])):
            return False
        
        elif isinstance(other, np.ndarray):
            return (np.allclose(self.c, other, rtol = rtol, atol = atol)
                and self.represents('Point', rtol = rtol, atol = atol))
        
        elif isinstance(other, Zonotope):
            # check center
            if not np.allclose(self.c, other.c, rtol = rtol, atol = atol):
                return False
            # compact both and compare generator matrices
            return comparison.compare_matrices(self.compact().G, other.compact().G,
                                               rtol = rtol, atol = atol, remove_zeros = True, check_negation = True)
        
        elif type(other).__name__ == 'Interval':
            return self.__eq__(Zonotope(**other.zonotope(mode = 'exact'), validate = False), rtol = rtol, atol = atol)
        
        elif type(other).__name__ in ['VPolytope', 'HPolyhedron']:
            return other.__eq__(self, rtol = rtol, atol = atol)

    # unary minus
    def __neg__(self) -> Zonotope:
        """Unary minus operator.

        Returns:
            Zonotope: Input Zonotope times -1.
        """
        return Zonotope(c = -self.c, G = self.G, validate=False)

    # unary plus
    def __pos__(self) -> Zonotope:
        """Unary plus operator.

        Returns:
            Zonotope: Same as input Zonotope.
        """
        return Zonotope(c = self.c, G = self.G, validate=False)

    # translation by vector
    def __sub__(self, other: np.ndarray) -> Zonotope:
        """Translation of a Zonotope by a vector.

        Args:
            other (np.ndarray): Vector.

        Raises:
            OtherFunctionError: If Zonotope - ConvexSet, call minkowski_difference instead.

        Returns:
            Zonotope: Result of the translation.
        """
        # TODO support int, float, list
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return Zonotope(c = self.c - other, G = self.G, validate = False)

        elif isinstance(other, ConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # point on boundary along a given direction
    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        """Computation of the point on the boundary of a Zonotope Z in a given direction starting from the zonotope center.

        Args:
            direction (np.ndarray): Direction along which to find the boundary point.

        Returns:
            np.ndarray: Boundary point.
        """
        self._checkOtherOperand(direction)

        # shift zonotope to origin
        Z = self - self.c
        return direction / Z.zonotope_norm(direction) + self.c
    
    # boundedness
    def bounded(self) -> bool:
        """Checks if a Zonotope Z is bounded.

        Returns:
            bool: Boundedness.
        """
        return True

    # Cartesian product
    def cartesian_product(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> Zonotope:
        """Cartesian product of a Zonotope Z and another ConvexSet or vector (np.ndarray) S.
        Defined as {[z^T s^T]^T | z in Z, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            Zonotope: Result of the Cartesian product.
        """
        self._checkMode(mode)

        n1 = self.dimension
        m1 = self.number_generators()

        if isinstance(other, np.ndarray):
            if m1 == 0:
                return Zonotope(c = np.hstack((self.c, other)), G = self.G, validate = False)
            else:
                return Zonotope(c = np.hstack((self.c, other)),
                                G = np.hstack((self.G, np.zeros((m1, other.size)))),
                                validate = False)

        elif isinstance(other, Zonotope):
            n2 = other.dimension
            m2 = other.number_generators()
            # concatenate centers
            center = np.hstack((self.c, other.c))
            # block-concatenate generator matrices
            generators = np.vstack((np.hstack((self.G, np.zeros((m1, n2)))),
                                    np.hstack((np.zeros((m2, n1)), other.G))))

            return Zonotope(c = center, G = generators, validate = False)

        else:
            # convert to zonotope and compute Cartesian product
            return self.cartesian_product(Zonotope(**other.zonotope(mode = mode), validate = False))

    # center
    def center(self) -> np.ndarray:
        """Center of a Zonotope Z.

        Returns:
            np.ndarray: Center of the Zonotope.
        """
        return self.c

    # compact representation
    def compact(self, *, rtol: float = 1e-12) -> Zonotope:
        """Minimal representation of a Zonotope Z.
        Unifies aligned generators and removes redundant generators.

        Args:
            rtol (float, optional): Relative tolerance for alignment check. Defaults to 1e-12.

        Returns:
            Zonotope: Zonotope in minimal representation.
        """
        # remove all-zero generators
        generators = self.G[np.any(self.G, axis=1), :]

        # check for aligned generators
        index_aligned = comparison.find_aligned_generators(generators, rtol=rtol)
        if index_aligned:
            # init array for new generators
            new_generators = np.zeros((len(index_aligned), self.dimension))
            for row, aligned_tuples in enumerate(index_aligned):
                # mask with factors 1 and -1 (invert direction if generators are anti-parallel)
                mask = np.logical_not(np.sign(generators[aligned_tuples, :])
                                      == np.reshape(np.sign(generators[aligned_tuples[0], :]), (1, self.dimension))) * -2. + 1.
                # add generators
                new_generators[row, :] = np.reshape(np.sum(generators[aligned_tuples, :] * mask, axis=0), (1, self.dimension))
            # replace aligned generators by new ones
            generators = np.vstack((np.delete(generators, index_aligned, axis=0), new_generators))

        return Zonotope(c = self.c, G = generators, validate = False)

    # containment check
    def contains(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks containment of a ConvexSet or vector (np.ndarray) S in a Zonotope Z.
        Defined as forall s in S: s in Z?

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Raises:
            NotImplementedError: Zonotope-in-zonotope not supported.

        Returns:
            bool: Containment status.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            if self.number_generators() == 0:
                return np.allclose(self.c, other, rtol = rtol, atol = atol)
            else:
                # shift zonotope and other by center of zonotope and check zonotope norm
                norm = (self - self.c).zonotope_norm(other - self.c)
                return norm <= 1 or np.isclose(norm, 1., rtol = rtol, atol = atol)
        elif isinstance(other, Zonotope) and other.number_generators() == 0:
            # shift zonotope and other by center of zonotope and check zonotope norm
            norm = (self - self.c).zonotope_norm(other.c - self.c)
            return norm <= 1 or np.isclose(norm, 1., rtol = rtol, atol = atol)
        else:
            # TODO: convert self to Hpolyhedron and use its contains function
            raise NotImplementedError

    # convex hull
    def convex_hull(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> Zonotope:
        """Convex hull of a Zonotope Z and another set or vector S.
        Defined as {lambda*z + (1-lambda)*s | z in Z, s in S, lambda in [0,1]}

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of operation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Modes 'inner' and 'exact' not supported in the general case.

        Returns:
            Zonotope: Result of the convex hull.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if mode in ['exact', 'inner']:
            # todo: implement special case 'single point - single point'
            raise NotImplementedError

        if not isinstance(other, Zonotope):
            return self.convex_hull(Zonotope(**other.zonotope(mode = mode), validate = False), mode = mode)

        # new center
        center = 0.5 * (self.c + other.c)

        # generator from centers
        generator_center = 0.5 * (self.c - other.c)

        # retrieve number of generators
        number_generators_self = self.number_generators()
        number_generators_other = other.number_generators()

        # new generator matrix
        if number_generators_self >= number_generators_other:
            generators = np.vstack((generator_center,
                                    0.5 * (self.G[:number_generators_other, :] + other.G),
                                    0.5 * (self.G[:number_generators_other, :] - other.G),
                                    self.G[number_generators_other:, :]))
        else:
            generators = np.vstack((generator_center,
                                    0.5 * (self.G + other.G[:number_generators_self, :]),
                                    0.5 * (self.G - other.G[:number_generators_self, :]),
                                    other.G[number_generators_self:, :]))

        return Zonotope(c = center, G = generators, validate = False)
    
    # degeneracy
    def degenerate(self, *, tol: float = 1e-12) -> bool:
        """Determines if a Zonotope Z is degenerate.

        Returns:
            bool: Degeneracy of the zonotope.
            rtol (float, optional): Tolerance. Defaults to 1e-12.
        """
        return self.number_generators() == 0 or np.linalg.matrix_rank(self.G, tol = tol) < self.dimension
    
    # emptiness
    def empty(self) -> bool:
        """Checks if a Zonotope Z is empty.

        Returns:
            bool: Emptiness.
        """
        return False

    # conversion to hpolyhedron
    def hpolyhedron(self, *, mode: str = 'exact') -> dict:
        """Conversion of a Zonotope Z to an HPolyhedron HP.

        Args:
            mode (str, optional): Approximation of conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: keyword arguments for instantiation of a HPolyhedron object
        """
        self._checkMode(mode)

        raise NotImplementedError
        # return {'A': A, 'b': b}

    # intersection check
    def intersects(self, other: Union[ConvexSet, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if a Zonotope Z intersects another set or vector S.
        Defined as exists s in Z: s in S?

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
        elif type(other).__name__ in ['VPolytope', 'HPolyhedron']:
            return other.intersects(self, rtol = rtol, atol = atol)
        elif type(other).__name__ == 'Interval':
            other = Zonotope(**other.zonotope(mode = 'exact'))

        # cases without generators: less intermediate computations
        if self.number_generators() == 0:
            return other.contains(self.c, rtol = rtol, atol = atol)
        elif other.number_generators() == 0:
            return self.contains(other.c, rtol = rtol, atol = atol)

        # use identity: Z1 intersects Z2 iff 0 in Z1 + (-Z2)
        return (self.minkowski_sum(-other)).contains(np.zeros(self.dimension), rtol = rtol, atol = atol)

    # conversion to interval
    def interval(self, *, mode: str = 'exact') -> dict:
        """Conversion to Interval.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Mode 'inner' only supported if the zonotope represents an interval.
            NotImplementedError: Mode 'exact' only supported if the zonotope represents an interval.

        Returns:
            dict: Keyword arguments for instantiation of an Interval object.
        """
        self._checkMode(mode)

        if mode == 'outer':
            # outer approximation
            radius = np.sum(np.abs(self.G), axis=0)
            lower_bound = self.c - radius
            upper_bound = self.c + radius
        elif mode == 'inner':
            # inner approximation or exact conversion
            if self.represents('Interval'):
                return self.interval(mode = 'outer')
            else:
                raise NotImplementedError
        elif mode == 'exact':
            # exact conversion (not always possible)
            if self.represents('Interval'):
                return self.interval(mode = 'outer')
            else:
                raise ExactEvaluationImpossibleError

        return {'lb': lower_bound, 'ub': upper_bound}

    # linear map
    def matmul(self, matrix: np.ndarray) -> Zonotope:
        """Linear map of a Zonotope Z by a matrix (np.ndarray) M.
        Defined as {M s | s in Z}.

        Args:
            matrix (np.ndarray): Matrix for left-multiplication.

        Returns:
            Zonotope: Result of the matrix multiplication.
        """
        self._checkMatrix(matrix)

        # linear transformation of center
        center = np.dot(matrix, self.c)
        if self.number_generators() == 0:
            return Zonotope(c = center, G = self.G, validate = False)

        # linear transformation of generator matrix
        generators = np.matmul(self.G, matrix.T)

        return Zonotope(c = center, G = generators, validate = False)

    # Minkowski sum
    def minkowski_sum(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> Zonotope:
        """Minkowski sum between a Zonotope Z and another set or vector S.
        Defined as {z + s | z in Z, s in S}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the evaluation: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            Zonotope: Result of the Minkowski sum.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        # Minkowski sum with...
        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return Zonotope(c = self.c + other, G = self.G, validate = False)

        elif isinstance(other, Zonotope):
            # ...a zonotope (exact computation possible)
            return Zonotope(c = self.c + other.c, G = np.vstack((self.G, other.G)), validate = False)

        else:
            # ...other ConvexSet object (convert to zonotope and then compute Minkowski sum)
            other = Zonotope(**other.zonotope(mode = mode), validate = False)
            return self.minkowski_sum(other)

    # Minkowski difference
    def minkowski_difference(self, other: Union[ConvexSet, np.ndarray], *, mode: str = 'exact') -> Zonotope:
        """Minkowski difference between a Zonotope Z and another set or vector S.
        Defined as {s | s + S in Z}.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Raises:
            NotImplementedError: Minkowski difference with a Zonotope as a subtrahend not supported.

        Returns:
            Zonotope: Result of the Minkowski difference.
        """
        self._checkOtherOperand(other)
        self._checkMode(mode)

        if isinstance(other, np.ndarray):
            return self - other

        raise NotImplementedError
    
    # number of generators
    def number_generators(self) -> int:
        """Returns the number of generators of a Zonotope Z. This is the number of rows in the generator matrix.

        Returns:
            int: Number of generators.
        """
        return self.G.shape[0]

    # projection onto subspace
    def project(self, *, axis: tuple) -> Zonotope:
        """Projection of a Zonotope Z onto a subspace.

        Args:
            axis (tuple): Subspace for projection.

        Returns:
            Zonotope: Projected Zonotope.
        """
        self._checkSubspace(axis)

        # convert tuples to lists for indexing
        center = self.c[list(axis)]
        generators = self.G[:, list(axis)] if self.G is not None else None
        return Zonotope(c = center, G = generators, validate = False)

    # zonotope order reduction (only Girard's method)
    def reduce(self, order: int) -> Zonotope:
        """Reduction of the set representation size of a Zonotope Z.
        Zonotope order reduction to an order greater or equal to 1.

        Args:
            order (int): Reduced order.

        Raises:
            ValueError: order must not be smaller than 1.

        Returns:
            Zonotope: Zonotope with reduced set representation size.
        """
        self_generators = self.number_generators()
        # exception handling
        if order < 1:
            raise ValueError('Zonotope:reduce',
                             'Order must be a number greater or equal to 1')

        # special cases
        elif self_generators == 0:
            # no generators -> no reduction
            return Zonotope(c = self.c, G = self.G, validate = False)
        elif order == 1:
            # corresponds to conversion to interval (unless fewer generators than self.dimension)
            if self_generators <= self.dimension:
                return Zonotope(c = self.c, G = self.G, validate = False)
            return Zonotope(c = self.c, G = np.diag(np.sum(np.abs(self.G), axis=0)), validate = False)
        elif order * self.dimension >= self_generators:
            # order is too large to cause any reduction
            return Zonotope(c = self.c, G = self.G)

        # compute number of remaining generators
        number_remaining_generators = int(np.floor(self.dimension * (order - 1)))
        number_reduced_generators = int(self_generators - number_remaining_generators)

        # compute Girard's metric for all generators
        girard_metric = np.linalg.norm(self.G, axis=1, ord=1) - np.linalg.norm(self.G, axis=1, ord=np.inf)

        # indices ascending in value of girard metric
        indices = np.argpartition(girard_metric, number_reduced_generators)

        # enclose selected generators by a box
        reduced_generators = np.diag(np.sum(np.abs(self.G[indices[:number_reduced_generators], :]), axis=0))

        return Zonotope(c = self.c,
                        G = np.vstack((self.G[indices[number_reduced_generators:], :], reduced_generators)),
                        validate = False)

    # representation by other set representation
    def represents(self, set_class: str, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if a Zonotope Z can also be equivalently represented using another ConvexSet class.

        Args:
            set_class (str): Name of another ConvexSet class or 'Point'.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: Representation possible.
        """
        self._checkSetClass(set_class)

        if set_class == 'Point':
            if self.number_generators() == 0:
                return True
            # compute size of box around generators
            interval_dict = self.interval(mode = 'outer')
            lower_bound, upper_bound = interval_dict['lb'], interval_dict['ub']
            return np.allclose(upper_bound - lower_bound, 0., rtol = rtol, atol = atol)
        
        if self.dimension == 1:
            return True
        
        if set_class == 'Interval':
            if self.number_generators() == 0:
                return True
            G_abs = np.abs(self.G)
            return np.allclose(np.sum(G_abs, axis=1), np.max(G_abs, axis=1), rtol = rtol, atol = atol)
        
        # every zonotope is a zonotope/hpolyhedron/vpolytope
        return True

    # support function evaluation
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Support function evaluation of a Zonotope Z in a direction d.
        Value defined as max_{s in Z} d^T * s.
        Vector defined as arg max_{s in Z} d^T * s.

        Args:
            direction (np.ndarray): Direction along which to evaluate the support function.

        Returns:
            tuple[float, np.ndarray]: Support value and support vector.
        """
        self._checkOtherOperand(direction)

        if self.number_generators() == 0:
            # no generators
            return (np.dot(direction, self.c), self.c)

        # auxiliary value: projected generator matrix
        G_projected = np.dot(self.G, direction)

        # value of support function
        value = np.dot(direction, self.c) + np.sum(np.abs(G_projected))

        # support vector
        factors = np.sign(G_projected)
        vector = self.c + np.dot(factors, self.G)

        # return value of support function and support vector
        return (value, vector)

    # vertex enumeration
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of a Zonotope Z.

        Returns:
            np.ndarray: 2D array containing vertices as rows.
        """
        # remove all-zero generators
        Z = self.compact()

        # init vertices by center
        V = np.reshape(self.c, (1, self.dimension))
        if Z.number_generators() == 0:
            return V
        # add first generator
        V = np.vstack((self.c + Z.G[0, :], self.c - Z.G[0, :]))

        # loop over all remaining generators
        for row in range(1, Z.number_generators()):
            # add next generator to all vertices
            V = np.vstack((V + Z.G[row, :], V - Z.G[row, :]))
            # compute convex hull and extract vertices
            V = V[ConvexHull(V).vertices, :]

        return V

    # volume computation
    def volume(self) -> float:
        """Volume computation of a Zonotope Z.
        Defined as: 2^n * sum_{nxn generator submatrices} |det(nxn generator submatrix)|
        Note: Degenerate zonotopes have a volume of zero.

        Returns:
            float: Volume.
        """
        # check degeneracy
        if self.degenerate():
            return 0.

        # lazy enumeration of all combinations of nxn submatrices
        all_combinations = combinations(range(self.number_generators()), r = self.dimension)

        vol = 0.
        for combination in all_combinations:
            vol = vol + np.abs(np.linalg.det(self.G[combination, :]))

        return 2**self.dimension * vol

    # conversion to vpolytope
    def vpolytope(self, *, mode: str = 'exact') -> dict:
        """Conversion to VPolytope.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of a VPolytope object.
        """
        self._checkMode(mode)

        return {'V': self.vertices()}

    # conversion to zonotope
    def zonotope(self, *, mode: str = 'exact') -> dict:
        """Overloaded conversion to Zonotope.

        Args:
            mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

        Returns:
            dict: Keyword arguments for instantiation of an Zonotope object.
        """
        self._checkMode(mode)

        return {'c': self.c, 'G': self.G}

    # zonotope norm
    def zonotope_norm(self, other: np.ndarray) -> float:
        """Computes the norm of a point with respect to the zonotope-norm induced by the zonotope Z.

        Args:
            other (np.ndarray): Vector.

        Raises:
            NotImplementedError: Center must be close to 0.

        Returns:
            float: Value of the zonotope norm.
        """
        self._checkOtherOperand(other)

        n, m = self.dimension, self.number_generators()
        # special case: no generators
        if m == 0:
            if np.allclose(other, 0):
                return 0.
            else:
                return np.inf

        # ensure that center is close to zero
        if not np.allclose(self.c, np.zeros(n)):
            raise NotImplementedError

        # objective function
        c = np.hstack((1., np.zeros(m)))

        # constraints
        A_eq = np.hstack((np.zeros((n, 1)), self.G.T))
        b_eq = other
        A_ub = np.vstack((np.hstack((-np.ones((m, 1)), np.eye(m))),
                          np.hstack((-np.ones((m, 1)), -np.eye(m)))))
        b_ub = np.zeros(2*m)

        # solve linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None))

        # objective function value at minimizer is the zonotope norm
        return res.fun
