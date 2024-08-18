from __future__ import annotations

from itertools import combinations
from typing import Union

import numpy as np
from scipy.optimize import linprog

from continuoussets.convexsets.interface_convexset import IConvexSet
from continuoussets.utils import comparison
from continuoussets.utils.exceptions import OtherFunctionError
from continuoussets.utils.auxiliary import number_singular_values, convex_hull

if __name__ == '__main__':
    print('This is the Zonotope class.')


class Zonotope(IConvexSet):

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

    # deep copy
    def copy(self) -> Zonotope:
        """Returns a deep copy of an Zonotope.

        Returns:
            Zonotope: Copied Zonotope.
        """
        return Zonotope(c = self.c.copy(), G = self.G.copy(), validate = False)

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
            OtherFunctionError: If other is a IConvexSet, call minkowski_sum instead.

        Returns:
            Zonotope: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return Zonotope(c = self.c + other, G = self.G, validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_sum')

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
            OtherFunctionError: If Zonotope - IConvexSet, call minkowski_difference instead.

        Returns:
            Zonotope: Result of the translation.
        """
        self._checkOtherOperand(other)

        if isinstance(other, np.ndarray):
            # ...a vector (exact computation possible)
            return Zonotope(c = self.c - other, G = self.G, validate = False)

        elif isinstance(other, IConvexSet):
            raise OtherFunctionError((self, other), 'minkowski_difference')

    # basis of the affine hull (for degenerate sets)
    def basis_affine_hull(self) -> tuple:
        """Computes a basis of the affine hull of a Zonotope Z.

        Returns:
            tuple: Matrix with basis vectors, number of basis vectors.
        """
        # use singular value decomposition
        # todo: use QR decomposition instead? (faster)
        matrix, S, _ = np.linalg.svd(np.matmul(self.G.T, self.G))
        s = number_singular_values(S)
        return (matrix, s)

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
        norm = Z.zonotope_norm(direction)
        if norm is None:
            return self.c
        return direction / norm + self.c
    
    # boundedness
    def bounded(self) -> bool:
        """Checks if a Zonotope Z is bounded.

        Returns:
            bool: Boundedness.
        """
        return True

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
        generators = self.G[np.any(self.G, axis = 1), :]

        # check for aligned generators
        index_aligned = comparison.find_aligned_generators(generators, rtol = rtol)
        if index_aligned:
            # init array for new generators
            new_generators = np.zeros((len(index_aligned), self.dimension))
            for row, aligned_tuples in enumerate(index_aligned):
                # mask with factors 1 and -1 (invert direction if generators are anti-parallel)
                mask = np.logical_not(np.sign(generators[aligned_tuples, :])
                                      == np.reshape(np.sign(generators[aligned_tuples[0], :]), (1, self.dimension))) * -2. + 1.
                # add generators
                new_generators[row, :] = np.reshape(np.sum(generators[aligned_tuples, :] * mask, axis = 0), (1, self.dimension))
            # replace aligned generators by new ones
            generators = np.vstack((np.delete(generators, index_aligned, axis = 0), new_generators))

        return Zonotope(c = self.c, G = generators, validate = False)
    
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

        # linear transformation of center and generator matrix
        center = np.dot(matrix, self.c)
        generators = np.matmul(self.G, matrix.T)
        return Zonotope(c = center, G = generators, validate = False)
    
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
    
    # projection onto its own affine hull
    def project_affine_hull(self) -> tuple:
        """Projects a zonotope onto its own affine hull.
        For degenerate zonotopes, the resulting zonotope is of lower dimension, but non-degenerate.

        Returns:
            tuple: Projected zonotope, projection matrix, center of the new coordinate system in the old coordinate system.
        """
        if not self.degenerate():
            return (self, np.eye(self.dimension), np.zeros(self.dimension))

        # compute basis of the affine hull and project zonotope onto it
        c = self.c
        M_proj, r = (self - c).basis_affine_hull()
        Z_proj = (self - c).matmul(M_proj.T)
        # remove flat dimensions
        # todo: check if one can also use center... (not zero everywhere...)
        non_flat = np.invert(np.all(np.isclose(Z_proj.G, 0.), axis = 0))
        Z_proj = Z_proj.project(axis = tuple(np.nonzero(non_flat)[0]))

        return (Z_proj, M_proj, c)

    # zonotope order reduction (only Girard's method)
    def reduce(self, order: int) -> Zonotope:
        """Reduction of the set representation size of a Zonotope Z.
        Zonotope order reduction to an order greater or equal to 1.

        Args:
            order (int): Reduced zonotope order.

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
            return Zonotope(c = self.c, G = np.diag(np.sum(np.abs(self.G), axis = 0)), validate = False)
        elif order * self.dimension >= self_generators:
            # order is too large to cause any reduction
            return Zonotope(c = self.c, G = self.G)

        # compute number of remaining generators
        number_remaining_generators = int(np.floor(self.dimension * (order - 1)))
        number_reduced_generators = int(self_generators - number_remaining_generators)

        # compute Girard's metric for all generators
        girard_metric = np.linalg.norm(self.G, axis = 1, ord=1) - np.linalg.norm(self.G, axis = 1, ord=np.inf)

        # indices ascending in value of girard metric
        indices = np.argpartition(girard_metric, number_reduced_generators)

        # enclose selected generators by a box
        reduced_generators = np.diag(np.sum(np.abs(self.G[indices[:number_reduced_generators], :]), axis = 0))

        return Zonotope(c = self.c,
                        G = np.vstack((self.G[indices[number_reduced_generators:], :], reduced_generators)),
                        validate = False)

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
        # ensure linearly independent generators
        Z = self.compact()
        n = Z.dimension

        # todo: check if necessary... ConvexHull cannot deal with 1D
        if n == 1:
            return np.vstack((Z.c + Z.G[0], Z.c - Z.G[0]))

        # init vertices by center, loop over remaining generators
        V = np.reshape(Z.c, (1, n))
        for row in range(Z.number_generators()):
            V = np.vstack((V + Z.G[row, :], V - Z.G[row, :]))
            # compute convex hull and extract vertices
            V = convex_hull(V)

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
