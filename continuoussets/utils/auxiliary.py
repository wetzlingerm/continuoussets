import numpy as np

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
