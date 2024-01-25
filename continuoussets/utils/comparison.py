import numpy as np

if __name__ == '__main__':
    "This is a utilities file for enhanced numerical comparison"


# comparison of two matrices
def compare_matrices(M1: np.ndarray, M2: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8,
                     remove_zeros: bool = False, check_negation: bool = False) -> bool:
    """Comparison of two matrices under certain conditions and tolerances.

    Args:
        M1 (np.ndarray): 2D matrix.
        M2 (np.ndarray): 2D matrix.
        rtol (float, optional): Relative tolerance. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance. Defaults to 1e-8.
        remove_zeros (bool, optional): Removal of all-zero columns. Defaults to False.
        check_negation (bool, optional): Columns are equivalent even if sign is different. Defaults to False.

    Returns:
        bool: Success of comparison.
    """
    # note: same default values for relative/absolute tolerance as in np.isclose

    # remove zeros from matrices
    if remove_zeros:
        M1 = M1[:, np.any(M1, axis=0)] if M1 is not None else None
        M2 = M2[:, np.any(M2, axis=0)] if M2 is not None else None

    # either both or none should have no entries
    M1_empty = (M1 is None or M1.shape[1] == 0)
    M2_empty = (M2 is None or M2.shape[1] == 0)
    if M1_empty and M2_empty:
        return True
    elif M1_empty != M2_empty:
        return False

    # check number of columns
    number_of_columns = M1.shape[1]
    if number_of_columns != M2.shape[1]:
        return False

    # index for columns in M2 (and M2_neg) that have been matched to a column in M1
    index_not_matched = np.full(number_of_columns, True)

    # loop over all columns in M1
    for M1_i in range(number_of_columns):
        # loop over all columns in M2
        for M2_j in range(number_of_columns):
            # only check columns that have not been matched
            if index_not_matched[M2_j]:
                # check for equality between the i-th column in M1 and the j-th column in M2
                if (np.all(np.isclose(M1[:, M1_i], M2[:, M2_j], rtol=rtol, atol=atol)) or
                        (check_negation and np.all(np.isclose(M1[:, M1_i], -M2[:, M2_j], rtol=rtol, atol=atol)))):
                    index_not_matched[M2_j] = False
                    break
        else:  # else-clause in for-else: loop terminated without break
            # -> no columns in M2 matched any columns in M1
            return False

    # all columns in M2 have been matched to a column in M1
    return True


def find_aligned_generators(M: np.ndarray, *, rtol: float = 1e-12) -> tuple[tuple]:
    """Computes the indices of aligned generators.

    Args:
        M (np.ndarray): 2D matrix.
        rtol (float, optional): Relative tolerance. Defaults to 1e-12.

    Returns:
        tuple[tuple]: Indices of aligned generators.
    """
    if M is None:
        return ()

    # normalize all columns
    number_of_columns = M.shape[1]
    M = M / np.linalg.norm(M, axis=0, ord=2)
    # init zero vector for comparison
    zero_vector = np.zeros(M.shape[0])

    # index for already matched columns
    index_not_matched = np.full(number_of_columns, True)
    # init list of indices as list
    list_of_indices = []

    # loop over all columns
    for M_i in range(number_of_columns):
        # init list for aligned columns
        start_new_sublist = True
        # loop over all columns
        for M_j in range(number_of_columns):
            # only check columns that have not been matched, avoid same indices
            if index_not_matched[M_j] and M_i != M_j:
                # columns must be parallel or anti parallel
                if (np.all(np.isclose(M[:, M_i], M[:, M_j], rtol=rtol)) or
                        np.all(np.isclose(M[:, M_i] + M[:, M_j], zero_vector, rtol=rtol))):
                    # start a new sublist
                    if start_new_sublist:
                        index_not_matched[M_i] = False
                        list_of_indices.append([M_i])
                        start_new_sublist = False
                    # append to list of aligned columns
                    index_not_matched[M_j] = False
                    list_of_indices[-1].append(M_j)

    # convert result to tuple
    return tuple(tuple(element) for element in list_of_indices)
