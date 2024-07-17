import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.vpolytope import VPolytope

class TestVPolytope(unittest.TestCase):

    def test_init(self):
        ''' Test for object instantation '''
        # cases:
        # - V: int
        # - V: float
        # - V: list int
        # - V: list float
        # - V: np.ndarray

        # init VPolytopes
        VP_int = VPolytope(V = 1)
        VP_float = VPolytope(V = -1.5)
        VP_list_int = VPolytope(V = [0, 1])
        VP_list_float = VPolytope(V = [0.5, 1.5])
        VP_np = VPolytope(V = np.array( [[0., 1.], [0.5, -1.0]] ))

        # check results
        assert np.array_equal(VP_int.V, np.array([[1.0]]))
        assert np.array_equal(VP_float.V, np.array([[-1.5]]))
        assert np.array_equal(VP_list_int.V, np.array( [[0.], [1.]] ))
        assert np.array_equal(VP_list_float.V, np.array( [[0.5], [1.5]] ))
        assert np.array_equal(VP_np.V, np.array( [[0., 1.], [0.5, -1.0]] ))

        # check exceptions
        with self.assertRaises(ValueError):
            # no input arguments provided
            VPolytope()
        with self.assertRaises(ValueError):
            # vertices are >2D
            VPolytope(V = np.array([[[1.0, -0.5], [2.0, 0.0]], [[0.5, 1.5], [-0.5, 1.0]]]))

    def test_repr(self):
        ''' Test for display on command window '''
        # cases:
        # - single vertex
        # - mutliple vertices

        # init vpolytopes
        V_singlevertex = np.array([1., 2., -1.])
        V_vertices = np.array([[1., 2., 2.], [-1., 0., -1.]])
        VP_1D = VPolytope(V = V_singlevertex)
        VP_2D = VPolytope(V = V_vertices)

        # check if commands run through (check output manually)
        print(VP_1D)
        print(VP_2D)
        assert True

    def test_add(self):
        ''' Test for positive translation '''
        # cases:
        # - vpolytope + vector

        # init vpolytope and vector
        V_singlevertex = np.array([4.0, -2.0])
        V_2D = np.array([[1., 2., 2.], [-1., 0., -1.]])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = V_2D)
        v = np.array([-2., 0.])

        # compute translation
        result1 = VP_1 + v
        result2 = VP_2 + v

        # manual computation
        true_result1 = VPolytope(V = np.array([2.0, -2.0]))
        true_result2 = VPolytope(V = np.array([[-1., 0., 0.], [-1., 0., -1.]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2

        # check exceptions
        with self.assertRaises(exceptions.OtherFunctionError):
            # call minkowski_sum instead of __add__
            VP_1 + VP_1

    def test_eq(self):
        ''' Test for set equality '''
        # cases:
        # - VPolytope x VPolytope

        V1 = np.array([[2.0, -1.0, 0.0], [1.0, -0.5, 0.5]])
        V2 = np.array([[-3.0, 1.0], [0.5, 1.0]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        assert VP_1 == VP_1
        assert not VP_1 == VP_2
   
if __name__ == '__main__':
    unittest.main()