import numpy as np
from itertools import product

if __name__ == '__main__':
    "This is a utilities file for auxiliary computations"


# halfspace representation for a single vector
def halfspace_representation_from_vector(v: np.ndarray) -> tuple:
    """Initialization of the halfspace representation from a single given vector.

    Args:
        v (np.ndarray): Vector.

    Returns:
        tuple: Set of inequalities A, b fulfilling Ax <= b for the given vector x.
    """
    n = v.size
    A = np.vstack((-np.ones(n), np.eye(n)))
    b = np.matmul(A, v)

    return (A, b)


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