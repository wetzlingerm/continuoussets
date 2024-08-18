import unittest
import numpy as np
from continuoussets.utils import auxiliary
from continuoussets.utils import comparison

from continuoussets import Interval


class TestAuxiliary(unittest.TestCase):

    def test_n_dim_cross_product(self):
        ''' Test for n-dimensional cross product '''
        # cases:
        # - 2D
        # - 3D
        M1 = np.array([[5.], [-2.]])
        M2 = np.array([[1., 2.], [3., 4.], [5., 6.]])

        result1 = auxiliary.n_dim_cross_product(M1)
        result2 = auxiliary.n_dim_cross_product(M2)

        true_result1 = np.array([-2., -5.])
        true_result2 = np.array([-2., 4., -2.])

        assert np.allclose(result1, true_result1)
        assert np.allclose(result2, true_result2)

    def test_remove_duplicate_points(self):
        ''' Test for removal of duplicate points in a list '''
        # cases:
        # - no duplicates
        # - duplicates
        # - duplicates up to tolerance
        M1 = np.array([[1., 0.], [-1., 1.], [1., 1.], [2., 1.]])
        M2 = np.array([[1., 0.], [-1., 1.], [1., 1.], [1., 0.], [2., 1.]])
        M3 = np.array([[1., 0.], [-1., 1.], [1., 1.], [1.001, 0.], [2., 1.]])

        result1 = auxiliary.remove_duplicate_points(M1)
        result2 = auxiliary.remove_duplicate_points(M2)
        result3 = auxiliary.remove_duplicate_points(M3)
        result4 = auxiliary.remove_duplicate_points(M3, atol = 0.01)

        assert np.array_equal(result1, M1)
        assert np.array_equal(result2, M1)
        assert np.array_equal(result3, M3)
        assert np.array_equal(result4, M1)

    def test_fourier_motzkin_elimination(self):
        ''' Test for Fourier-Motzkin elimination '''
        # cases:
        # - 4D -> 3D
        A = np.array([[1., 2., 0., -1.], [1., -1., -1., 0.], [0., 1., 2., 2.],
                      [-2., -1., 0., 0.], [-1., 1., 1., 0.], [0., 0., 1., 2.],
                      [0., -3., 2., 1.], [1., 0., 0., -1.], [3., -1., -2., -1.]])
        b = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

        A_proj1, b_proj1 = auxiliary.fourier_motzkin_elimination(A, b, 0)
        A_proj2, b_proj2 = auxiliary.fourier_motzkin_elimination(A, b, 1)
        A_proj3, b_proj3 = auxiliary.fourier_motzkin_elimination(A, b, 2)
        A_proj4, b_proj4 = auxiliary.fourier_motzkin_elimination(A, b, 3)

        true_result_A_proj1 = np.array([[1., 2., 2.], [0., 1., 2.], [-3., 2., 1.], [3., 0., -2.],
                                        [-3., -2., 0.], [-1., 0., -2.], [-5., -4., -2.],
                                        [3., 1., -1.], [0., 0., 0.], [1., 1., -1.], [2., 1., -1.]])
        true_result_b_proj1 = np.array([1., 1., 1., 3., 3., 3., 5., 2., 2., 2., 4.])
        true_result_A_proj2 = np.array([[0., 1., 2.], [1., 0., -1.], [3., -2., -1.], [1., 1., 2.],
                                        [0., 0., 0.], [-3., 0., -1.], [-2., 2., 2.], [-3., 1., 0.],
                                        [3., 4., -1.], [0., 8., 7.], [-3., 5., 1.], [7., -4., -3.],
                                        [3., 0., 1.], [2., -1., -1.]])
        true_result_b_proj2 = np.array([1., 1., 3., 2., 2., 3., 2., 2., 5., 4., 4., 3., 2., 2.])
        true_result_A_proj3 = np.array([[1., 2., -1.], [-2., -1., 0.], [1., 0., -1.], [2., -1., 2.],
                                        [0., 0., 0.], [1., -1., 2.], [2., -5., 1.], [6., 0., 2.],
                                        [1., 1., -1.], [3., -1., 3.], [6., -8., 0.]])
        true_result_b_proj3 = np.array([1., 1., 1., 3., 2., 2., 3., 4., 3., 3., 4.])
        true_result_A_proj4 = np.array([[1., -1., -1.], [-2., -1., 0.], [-1., 1., 1.], [2., 5., 2.],
                                        [2., 4., 1.], [1., -1., 2.], [2., 1., 2.], [2., 0., 1.],
                                        [1., -3., 2.], [6., -1., -2.], [6., -2., -3.], [3., -4., 0.]])
        true_result_b_proj4 = np.array([1., 1., 1., 3., 3., 2., 3., 3., 2., 3., 3., 2.])

        assert comparison.compare_matrices(b_proj1, true_result_b_proj1)
        assert comparison.compare_matrices(A_proj1, true_result_A_proj1)
        assert comparison.compare_matrices(A_proj2, true_result_A_proj2)
        assert comparison.compare_matrices(b_proj2, true_result_b_proj2)
        assert comparison.compare_matrices(A_proj3, true_result_A_proj3)
        assert comparison.compare_matrices(b_proj3, true_result_b_proj3)
        assert comparison.compare_matrices(A_proj4, true_result_A_proj4)
        assert comparison.compare_matrices(b_proj4, true_result_b_proj4)

    def test_number_singular_values(self):
        ''' Test for number of singular values '''
        # cases:
        # - all values non-zero
        # - some values exactly zero
        # - some values near zero
        S1 = np.array([2., 1., 0.1])
        S2 = np.array([2., 1., 0.1, 0.])
        S3 = np.array([2., 1., 0.1, 0.00001])

        assert 3 == auxiliary.number_singular_values(S1)
        assert 3 == auxiliary.number_singular_values(S2)
        assert 4 == auxiliary.number_singular_values(S3, atol = 0., rtol = 0.)
        assert 3 == auxiliary.number_singular_values(S3, atol = 0.0001)

    def test_active_inequality(self):
        ''' Test for check of active inequalities '''
        # cases:
        # - no active
        # - active, non-degenerate
        # - active because degenerate
        A_nondeg = np.array([[1., 0.], [-1., 1.], [-1., -1.]])
        b_nondeg = np.array([1., 1., 1.])
        A_deg = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        b_deg = np.array([1., 0., 1., 0.])

        assert not auxiliary.active_inequality(A_nondeg, b_nondeg, np.array([0., 0.]))
        assert auxiliary.active_inequality(A_nondeg, b_nondeg, np.array([1., 1.]))
        assert auxiliary.active_inequality(A_deg, b_deg, np.array([0.5, 0.5]))

    def test_convex_hull(self):
        ''' Test for convex hull of degenerate and one-dimensional point clouds '''
        # cases:
        # - single point
        # - 1D
        # - non-degenerate
        # - degenerate
        V_point = np.array([[1., 0., -1.]])
        V_1D = np.array([[-2.], [0.], [-1.], [4.], [6.]])
        V_1D_single = np.array([[-2.], [-2.]])
        V_nondeg = np.array([[1., 0.], [0., 1.], [-1., -1.]])
        V_deg = np.array([[1., 0., 1.], [0., 1., 1.], [-1., -1., -2.], [0., 0., 0.]])

        result_point = auxiliary.convex_hull(V_point)
        result_1D = auxiliary.convex_hull(V_1D)
        result_1D_single = auxiliary.convex_hull(V_1D_single)
        result_nondeg = auxiliary.convex_hull(V_nondeg)
        result_deg = auxiliary.convex_hull(V_deg)

        true_result_point = V_point
        true_result_1D = np.array([[-2.], [6.]])
        true_result_1D_single = np.array([[-2.]])
        true_result_nondeg = V_nondeg
        true_result_deg = np.array([[1., 0., 1.], [0., 1., 1.], [-1., -1., -2.]])

        assert comparison.compare_matrices(result_point, true_result_point)
        assert comparison.compare_matrices(result_1D, true_result_1D)
        assert comparison.compare_matrices(result_1D_single, true_result_1D_single)
        assert comparison.compare_matrices(result_nondeg, true_result_nondeg)
        assert comparison.compare_matrices(result_deg, true_result_deg)

    def test_SetPair(self):
        ''' Test for SetPair class '''
        # __init__
        I1 = Interval(lb = np.array([-1.]), ub = np.array([4.]))
        s = auxiliary.SetPair(I1, 'Vector')
        # __repr__
        print(s)
        # __eq__
        assert not s == 5
        

if __name__ == '__main__':
    unittest.main()