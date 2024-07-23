import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.interval import Interval

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
        assert VP_int.dimension == 1
        assert np.array_equal(VP_float.V, np.array([[-1.5]]))
        assert VP_float.dimension == 1
        assert np.array_equal(VP_list_int.V, np.array( [[0., 1.]] ))
        assert VP_list_int.dimension == 1
        assert np.array_equal(VP_list_float.V, np.array( [[0.5, 1.5]] ))
        assert VP_list_float.dimension == 1
        assert np.array_equal(VP_np.V, np.array( [[0., 1.], [0.5, -1.0]] ))
        assert VP_np.dimension == 2

        # check exceptions
        with self.assertRaises(ValueError):
            # no input arguments provided
            VPolytope()
        with self.assertRaises(ValueError):
            # vertices are >2D
            VPolytope(V = np.array([[[1.0, -0.5], [2.0, 0.0]], [[0.5, 1.5], [-0.5, 1.0]]]))
        with self.assertRaises(TypeError):
            # vertices are of wrong type
            VPolytope(V = "vertex")

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
        V_singlevertex = np.array([[4.0], [-2.0]])
        V_2D = np.array([[1., 2., 2.], [-1., 0., -1.]])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = V_2D)
        v = np.array([-2., 0.])

        # compute translation
        result1 = VP_1 + v
        result2 = VP_2 + v

        # manual computation
        true_result1 = VPolytope(V = np.array([[2.0], [-2.0]]))
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
        # - VPolytope x Interval

        V1 = np.array([[2.0, -1.0, 0.0], [1.0, -0.5, 0.5]])
        V2 = np.array([[-3.0, 1.0], [0.5, 1.0]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        I = Interval(lb = np.array([-2., -1.]), ub = np.array([4., 0.]))
        V_I = np.array([[-2., -2., 4., 4.], [-1., 0., -1., 0.]])
        VP_I = VPolytope(V = V_I)

        assert VP_1 == VP_1
        assert not VP_1 == VP_2
        assert VP_I == I

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[1.], [3.]])
        V2 = np.array([[2.0, -1.0, 0.0], [1.0, -0.5, 0.5]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        assert np.array_equal((-VP_1).V, -V1)
        assert np.array_equal((-VP_2).V, -V2)

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[1.], [3.]])
        V2 = np.array([[2.0, -1.0, 0.0], [1.0, -0.5, 0.5]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        assert np.array_equal((+VP_1).V, V1)
        assert np.array_equal((+VP_2).V, V2)

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - vpolytope - vector

        # init vpolytope and vector
        V_singlevertex = np.array([[4.0], [-2.0]])
        V_2D = np.array([[1., 2., 2.], [-1., 0., -1.]])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = V_2D)
        v = np.array([-2., 0.])

        # compute translation
        result1 = VP_1 - v
        result2 = VP_2 - v

        # manual computation
        true_result1 = VPolytope(V = np.array([[6.0], [-2.0]]))
        true_result2 = VPolytope(V = np.array([[3., 4., 4.], [-1., 0., -1.]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2

        # check exceptions
        with self.assertRaises(exceptions.OtherFunctionError):
            # call minkowski_difference instead of __sub__
            VP_1 - VP_1

    def test_boundary_point(self):
        ''' Test for boundary point computation '''
        # cases:
        # - vpolytope: single vertex

        V_singlevertex = np.array([[4.0], [-2.0]])
        VP_singlevertex = VPolytope(V = V_singlevertex)
        direction = np.array([-1., 0.])

        with self.assertRaises(NotImplementedError):
            VP_singlevertex.boundary_point(direction)

    def test_cartesian_product(self):
        ''' Test for Cartesian product '''
        # cases:
        # - vpolytope x vpolytope
        # - vpolytope x interval
        V_1 = np.array([1., 2., 4.])
        V_2 = np.array([-3., 5.])
        VP_1 = VPolytope(V = V_1)
        VP_2 = VPolytope(V = V_2)
        I_2 = Interval(lb = -3, ub = 5)

        V1_V2 = np.array([[1., 1., 4., 4.], [-3., 5., -3., 5.]])

        VP1_VP2 = VP_1.cartesian_product(VP_2)
        VP1_VP2 = VP1_VP2.compact()
        VP1_I2 = VP_1.cartesian_product(I_2)
        VP1_I2 = VP1_I2.compact()

        assert comparison.compare_matrices(VP1_VP2.V, V1_V2)
        assert comparison.compare_matrices(VP1_I2.V, V1_V2)

    def test_center(self):
        ''' Test for computation of center '''
        # cases:
        # - single vertex
        # - multiple vertices
        V_singlevertex = np.array([[4.], [-2.]])
        V_2D = np.array([[1., 2., 2.], [-1., 0., -1.]])

        VP_singlevertex = VPolytope(V = V_singlevertex)
        VP_2D = VPolytope(V = V_2D)

        c_single = VP_singlevertex.center()
        assert np.array_equal(c_single, V_singlevertex)

        with self.assertRaises(NotImplementedError):
            c_2D = VP_2D.center()

    def test_compact(self):
        ''' Test for minimal representation '''
        # cases:
        # - single vertex
        # - degenerate vertices #todo
        # - multiple vertices (no redundancies)
        # - multiple vertices (with redundancies)
        V_singlevertex = np.array([[4.], [-2.]])
        VP_singlevertex = VPolytope(V = V_singlevertex)

        V_multiple_no_red = np.array([[1., 0., -1., 0., 1.], [0., 1., 0., -1., -1.]])
        VP_multiple_no_red = VPolytope(V = V_multiple_no_red)

        V_multiple_red = np.array([[1., 0., -1., 0., 0., 1., 0.2, -0.3], [0., 1., 0., -0.5, -1., -1., 0.3, -0.1]])
        VP_multiple_red = VPolytope(V = V_multiple_red)

        result_singlevertex = VP_singlevertex.compact()
        result_multiple_no_red = VP_multiple_no_red.compact()
        result_multiple_red = VP_multiple_red.compact()

        assert np.array_equal(V_singlevertex, result_singlevertex.V)
        assert comparison.compare_matrices(V_multiple_no_red, result_multiple_no_red.V)
        assert comparison.compare_matrices(V_multiple_no_red, result_multiple_red.V)

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - vpolytope x vector
        V_2D = np.array([[1., -2., 0.], [-1., 0., 1.]])
        VP_2D = VPolytope(V = V_2D)
        v = np.array([0., 0.])

        assert VP_2D.contains(v)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # todo
        assert True
    
    def test_intersects(self):
        ''' Test for intersection check '''
        # todo
        assert True
    
    def test_interval(self):
        ''' Test for interval conversion '''
        # todo
        assert True
    
    def test_matmul(self):
        ''' Test for linear map '''
        # todo
        assert True
    
    def test_minkowski_sum(self):
        ''' Test for Minkowski sum '''
        # todo
        assert True

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - vpolytope - vector
        # - vpoltyope - vpolytope
        # todo
        assert True

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[2.], [3.], [5.], [-1.]])
        V2 = np.array([[-1., 0., 0., 1., 1., 2.],
                       [0., -1., 1., -1., 1., 2.],
                       [-1., 0., 2., 1., 1., 0.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        VP_1_proj = VPolytope(V = np.array([[2.], [-1.]]))
        VP_2_proj = VPolytope(V = np.array([[0., -1., 1., -1., 1., 2.],
                                            [-1., 0., 2., 1., 1., 0.]]))

        assert VP_1_proj == VP_1.project(axis = (0,3))
        assert VP_2_proj == VP_2.project(axis = (1,2))

    def test_represents(self):
        ''' Test for representation check '''
        # todo
        assert True

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - single vertex
        # - multiple vertices incl. redundancies
        V1 = np.array([[2.], [3.]])
        V2 = np.array([[-1., 0., 0., 0., 1., 1., 2., 2.], [0., -1., 0., 1., -1., 1., 0., 2.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        value1, vector1 = VP_1.support_function(np.array([1., 0.]))
        value2, vector2 = VP_2.support_function(np.array([1., 1.]))

        assert value1 == 2 and np.array_equal(vector1, np.array([[2.], [3.]]))
        assert value2 == 4 and np.array_equal(vector2, np.array([[2.], [2.]]))

    def test_vertices(self):
        ''' Test for vertex enumeration '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2.], [3.], [-1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., -1., 0.], [1., 2., -4.]])
        VP_2 = VPolytope(V = V2)

        assert comparison.compare_matrices(VP_1.vertices(), V1)
        assert comparison.compare_matrices(VP_2.vertices(), V2)

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2.], [3.], [-1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., -1., 0.], [1., 2., -4.]])
        VP_2 = VPolytope(V = V2)

        assert VP_1.volume() == 0
        with self.assertRaises(NotImplementedError):
            VP_2.volume()

    def test_vpolytope(self):
        ''' Test for overloaded conversion to vpolytope '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2.], [3.], [-1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., -1., 0.], [1., 2., -4.]])
        VP_2 = VPolytope(V = V2)

        result1 = VP_1.vpolytope()
        result2 = VP_2.vpolytope()

        assert comparison.compare_matrices(result1['V'], V1)
        assert comparison.compare_matrices(result2['V'], V2)

    def test_zonotope(self):
        ''' Test for zonotope conversion '''
        # todo
        assert True

if __name__ == '__main__':
    unittest.main()