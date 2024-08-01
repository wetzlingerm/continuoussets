import unittest
import numpy as np
from continuoussets.utils import auxiliary
from continuoussets.convexsets.hpolyhedron import HPolyhedron

class TestAuxiliary(unittest.TestCase):

    def test_halfspace_representation_from_vector(self):
        ''' Test for generation of halfspace representation from vector '''
        # cases:
        # - 1D
        # - 2D
        # - 5D
        v_1D = np.array([2.])
        v_2D = np.array([-1., 2.])
        v_5D = np.array([5., 3., -2., 1., 0.])
        A_1D, b_1D = auxiliary.halfspace_representation_from_vector(v_1D)
        HP_1D = HPolyhedron(A = A_1D, b = b_1D)
        A_2D, b_2D = auxiliary.halfspace_representation_from_vector(v_2D)
        HP_2D = HPolyhedron(A = A_2D, b = b_2D)
        A_5D, b_5D = auxiliary.halfspace_representation_from_vector(v_5D)
        HP_5D = HPolyhedron(A = A_5D, b = b_5D)

        assert HP_1D.contains(v_1D)
        assert HP_1D.degenerate()
        assert HP_2D.contains(v_2D)
        assert HP_2D.degenerate()
        assert HP_5D.contains(v_5D)
        assert HP_5D.degenerate()

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

if __name__ == '__main__':
    unittest.main()