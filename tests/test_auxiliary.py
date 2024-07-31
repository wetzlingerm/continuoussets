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

if __name__ == '__main__':
    unittest.main()