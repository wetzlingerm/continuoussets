from __future__ import annotations
from inspect import isclass

from typing import Union, Tuple, Type, Callable, Any

import numpy as np
from itertools import product
from scipy.spatial import ConvexHull
from scipy.linalg import svd

if __name__ == '__main__':
    "This is a utilities file for auxiliary computations"
    

# pair of sets
class SetPair:

    def __init__(self, first_operand: str, second_operand: str) -> SetPair:
        def read_name(operand: Union[Type, str]) -> str:
            if isinstance(operand, str):
                name = operand
            elif isclass(type(operand)):
                name = operand.__class__.__name__
            else:
                raise TypeError
            
            # unify names for points
            if name in ['ndarray', 'Point', 'Vector']:
                name = 'Point'
            return name
        
        self.sets = (read_name(first_operand), read_name(second_operand))

    def __repr__(self) -> str:
        return f"({self.sets[0]}, {self.sets[1]})"
        
    def __eq__(self, other: SetPair) -> bool:
        if not isinstance(other, SetPair):
            return False
        return self.sets == other.sets

    def __hash__(self) -> int:
        return hash(self.sets)


# decorator adding a map from input types to functions
def StrategyRegistry(cls):

    # add class variables
    setattr(cls, 'strategies', dict())

    # add functions operating on these class variables depending on member variables of class
    def register_strategy(pair: Union[Any, Tuple[Any]]) -> Callable:
        def decorator(func: Callable):
            # throw error if strategy has already been registered
            def update_strategies(new_key):
                if cls.strategies.get(new_key) is not None:
                    raise KeyError
                cls.strategies.update({new_key: func})

            if isinstance(pair, Tuple):
                [update_strategies(i_pair) for i_pair in pair]
            else:
                update_strategies(pair)
            return func
        return decorator
    
    @classmethod
    def select_strategy(cls, key: Any) -> Callable:
        return cls.strategies.get(key)

    # add functions to classs
    setattr(cls, 'register_strategy', register_strategy)
    setattr(cls, 'select_strategy', select_strategy)

    return cls
    

# n-dimensional cross product
def n_dim_cross_product(M: np.ndarray) -> np.ndarray:
    """Computes the n-dimensional cross product of a matrix.

    Args:
        M (np.ndarray): Matrix with n rows and (n-1) columns.

    Returns:
        np.ndarray: Result of the operation, vector with n entries.
    """
    n = M.shape[0]
    result = np.zeros(n)
    index = np.full(n, True)
    for i in range(n):
        # remove ith row and compute value (note that exponent is i+2 due to 0-indexing/1-indexing)
        index[i] = False
        result[i] = (-1)**(i+2) * np.linalg.det(M[index])
        index[i] = True

    return result


# remove duplicate vectors in a matrix
def remove_duplicate_points(M: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> np.ndarray:
    """Removes duplicate points from a list of points up to a given relative and absolute tolerance.

    Args:
        M (np.ndarray): Matrix with points as rows.
        rtol (float, optional): Relative tolerance. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance. Defaults to 1e-8.

    Returns:
        np.ndarray: Matrix without duplicates up to given tolerance.
    """
    # store indices for points that are kept
    m = M.shape[0]
    indices_keep = np.full(m, True)

    for i in range(m):
        # only check if a point has not been excluded prior
        if indices_keep[i]:
            # check for similarity up to tolerance using the difference to the ith point
            same = np.all(np.isclose(M - M[i], 0., rtol = rtol, atol = atol), axis = 1)
            # since any point is equal to itself, there has to be more than one occurrence
            if np.count_nonzero(same) > 1:
                # keep one copy, remove all the others
                same[i] = False
                indices_keep = np.logical_and(indices_keep, np.invert(same))

    return M[indices_keep]


# Fourier-Motzkin elimination
def fourier_motzkin_elimination(A: np.ndarray, b: np.ndarray, i: int) -> tuple:
    """Fourier-Motzkin elimination: projects a set of inequality Ax <= b onto dimension i.

    Args:
        A (np.ndarray): Constraint matrix p x n.
        b (np.ndarray): Constraint offset p x 1.
        i (int): Dimension.

    Returns:
        tuple: Projected matrix p x (n-1) and offset p x 1.
    """
    # for stability, set all values closer than a given tolerance to exactly 0
    atol = 1e-8
    A_proj = A.copy()
    A_proj[abs(A_proj) < atol] = 0.

    # divide the i-th column into positive, zero, and negative entries
    Z, = np.nonzero(A_proj[:, i] == 0)
    P, = np.nonzero(A_proj[:, i] > 0)
    N, = np.nonzero(A_proj[:, i] < 0)

    # Cartesian product N x P
    p = product(N, P)

    # init projection matrix
    U = np.zeros((Z.size + P.size*N.size, b.size))
    # deal with Z
    for j, dim_i in enumerate(Z):
        U[j, dim_i] = 1

    # deal with N x P
    for j, pair in enumerate(p):
        U[Z.size + j, pair[0]] = A_proj[pair[1], i]
        U[Z.size + j, pair[1]] = -A_proj[pair[0], i]

    # projection
    A_proj = np.matmul(U, A_proj)
    A_proj = np.delete(A_proj, i, 1)
    b_proj = np.matmul(U, b)

    return (A_proj, b_proj)


# number of singular values
def number_singular_values(S: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-12) -> int:
    """Returns the number of non-zero singular values, up to a given absolute tolerance.

    Args:
        S (np.ndarray): Matrix with singular values, resulting from U,S,V = np.linalg.svd(M)
        rtol (float, optional): Relative tolerance. Defaults to 1e-10.
        atol (float, optional): Absolute tolerance. Defaults to 1e-12.

    Returns:
        int: Number of non-zero singular values.
    """
    return np.nonzero(np.invert(np.isclose(S, 0., rtol = rtol, atol = atol)))[0].size


# check if any inequality is active for a given point
def active_inequality(A: np.ndarray, b: np.ndarray, x: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    """Checks if there is an active inequality in set Ax <= b for a given vector x.
    Note: This function does not check whether x is contained in {x | Ax <= b}.

    Args:
        A (np.ndarray): Constraint matrix (2D).
        b (np.ndarray): Constraint offset (1D).
        x (np.ndarray): Vector (1D).
        rtol (float, optional): Relative tolerance. Defaults to 1e-10.
        atol (float, optional): Absolute tolerance. Defaults to 1e-12.

    Returns:
        bool: Status of active inequalities.
    """
    return np.any(np.isclose(b - np.matmul(A, x), 0., rtol = rtol, atol = atol))


# sort rows
def sort_rows(A: np.ndarray) -> np.ndarray:
    """Sorts a matrix A row-wise.

    Args:
        A (np.ndarray): 2D matrix.

    Returns:
        np.ndarray: Sorted matrix.
    """
    return A[np.lexsort(A.T[::-1])]


# convex hull for degenerate cases
def convex_hull(V: np.ndarray) -> np.ndarray:
    """Computes the convex hull of a set of a potentially degenerate or one-dimensional vertices.
    Made necessary since scipy.spatial.ConvexHull cannot deal with these additional cases.

    Args:
        V (np.ndarray): 2D matrix with vertices.

    Returns:
        np.ndarray: Convex hull of vertices.
    """
    # helper function for 1D case
    def convex_hull_1D(V):
        V_min = np.min(V, axis = 0)
        V_max = np.max(V, axis = 0)
        if np.isclose(V_min, V_max):
            return np.reshape(V_min, (1, ))
        return np.vstack((V_min, V_max))

    # we have m vertices of dimension n
    m, n = V.shape
    if m == 1:
        return V.copy()
    elif n == 1:
        return convex_hull_1D(V)

    # we want to use the ConvexHull function... special handling for degenerate sets
    try:
        V = V[ConvexHull(V).vertices, :]
    except Exception:
        # one of multiple cases:
        # 1. not enough points(<=n) to construct initial simplex (need n+1)
        # 2. Initial simplex is flat (facet k is coplanar with the interior point)
        # ...project onto its affine hull and compute vertices there, then project back
        c = np.mean(V, axis = 0)
        V_shifted = (V - c).T
        U, S, _ = svd(V_shifted)
        r = number_singular_values(S)

        V_subspace = np.matmul(np.matmul(np.hstack((np.eye(r), np.zeros((r, n-r)))), U.T), V_shifted)
        if V_subspace.shape[0] == 1:  # 1D
            V_subspace = convex_hull_1D(V_subspace.T).T
        else:
            V_subspace = V_subspace[:, ConvexHull(V_subspace.T).vertices]
        V = np.matmul(np.matmul(U, np.vstack((np.eye(r), np.zeros((n-r, r))))), V_subspace).T + c

    return V
