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
