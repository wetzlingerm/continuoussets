import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.interval import Interval
from continuoussets.convexsets.zonotope import Zonotope

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
        assert VP_list_int.dimension == 2
        assert np.array_equal(VP_list_float.V, np.array( [[0.5, 1.5]] ))
        assert VP_list_float.dimension == 2
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
        V_singlevertex = np.array([[4.0, -2.0]])
        V_2D = np.array([[1., -1.], [2., 0.], [2., -1.]])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = V_2D)
        v = np.array([-2., 0.])

        # compute translation
        result1 = VP_1 + v
        result2 = VP_2 + v

        # manual computation
        true_result1 = VPolytope(V = np.array([[2.0, -2.0]]))
        true_result2 = VPolytope(V = np.array([[-1., -1.], [0., 0.], [0., -1.]]))

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

        V1 = np.array([[2., 1.], [-1., -0.5], [0., 0.5]])
        V2 = np.array([[-3., 0.5], [1., 1.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        I = Interval(lb = np.array([-2., -1.]), ub = np.array([4., 0.]))
        V_I = np.array([[-2., -1.], [-2., 0.], [4., -1.], [4., 0.]])
        VP_I = VPolytope(V = V_I)

        assert VP_1 == VP_1
        assert not VP_1 == VP_2
        assert VP_I == I

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[1., 3.]])
        V2 = np.array([[2., 1.], [-1., -0.5], [0., 0.5]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        assert np.array_equal((-VP_1).V, -V1)
        assert np.array_equal((-VP_2).V, -V2)

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[1., 3.]])
        V2 = np.array([[2., 1.], [-1., -0.5], [0., 0.5]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        assert np.array_equal((+VP_1).V, V1)
        assert np.array_equal((+VP_2).V, V2)

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - vpolytope - vector

        # init vpolytope and vector
        V_singlevertex = np.array([[4., -2.]])
        V_2D = np.array([[1., -1.], [2., 0.], [2., -1.]])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = V_2D)
        v = np.array([-2., 0.])

        # compute translation
        result1 = VP_1 - v
        result2 = VP_2 - v

        # manual computation
        true_result1 = VPolytope(V = np.array([[6., -2.]]))
        true_result2 = VPolytope(V = np.array([[3., -1.], [4., 0.], [4., -1.]]))

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

        V_singlevertex = np.array([[4., -2.]])
        VP_singlevertex = VPolytope(V = V_singlevertex)
        direction = np.array([-1., 0.])

        with self.assertRaises(NotImplementedError):
            VP_singlevertex.boundary_point(direction)

    def test_cartesian_product(self):
        ''' Test for Cartesian product '''
        # cases:
        # - vpolytope x vpolytope
        # - vpolytope x interval
        V_1 = np.array([[1.], [2.], [4.]])
        V_2 = np.array([[-3.], [5.]])
        VP_1 = VPolytope(V = V_1)
        VP_2 = VPolytope(V = V_2)
        I_2 = Interval(lb = -3, ub = 5)

        V1_V2 = np.array([[1., -3.], [1., 5.], [4., -3.], [4., 5.]])

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
        V_singlevertex = np.array([[4., -2.]])
        V_2D = np.array([[1., -1.], [2., 0.], [2., -1.]])

        VP_singlevertex = VPolytope(V = V_singlevertex)
        VP_2D = VPolytope(V = V_2D)

        c_single = VP_singlevertex.center()
        c_2D = VP_2D.center()

        assert np.array_equal(c_single, np.reshape(V_singlevertex, (2,)))
        assert VP_2D.contains(c_2D)

    def test_compact(self):
        ''' Test for minimal representation '''
        # cases:
        # - single vertex
        # - one-dimensional
        # - degenerate vertices #todo
        # - multiple vertices (no redundancies)
        # - multiple vertices (with redundancies)
        V_singlevertex = np.array([[4., -2.]])
        VP_singlevertex = VPolytope(V = V_singlevertex)

        V_multiple_no_red = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., -1.]])
        VP_multiple_no_red = VPolytope(V = V_multiple_no_red)

        V_multiple_red = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -0.5], [0., -1.], [1., -1.], [0.2, -0.3], [-0.3, -0.1]])
        VP_multiple_red = VPolytope(V = V_multiple_red)

        V_1D = np.array([[1.], [-1.], [0.], [2.]])
        VP_1D = VPolytope(V = V_1D)

        result_singlevertex = VP_singlevertex.compact()
        result_multiple_no_red = VP_multiple_no_red.compact()
        result_multiple_red = VP_multiple_red.compact()
        result_1D = VP_1D.compact()

        assert np.array_equal(V_singlevertex, result_singlevertex.V)
        assert comparison.compare_matrices(V_multiple_no_red, result_multiple_no_red.V)
        assert comparison.compare_matrices(V_multiple_no_red, result_multiple_red.V)
        assert comparison.compare_matrices(np.array([[-1.], [2.]]), result_1D.V)

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - vpolytope x vector
        # - vpolytope x vpolytope (self)
        # - vpolytope x interval
        V_2D = np.array([[1., -1.], [-2., 0.], [0., 1.]])
        VP_2D = VPolytope(V = V_2D)
        v = np.array([0., 0.])
        I_2D = Interval(lb = np.array([-0.1, -0.1]), ub = np.array([0.1, 0.1]))

        assert VP_2D.contains(v)
        assert VP_2D.contains(VP_2D)
        assert VP_2D.contains(I_2D)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # cases:
        # - single vertex x single vertex
        # - vpolytope x vpolytope
        # - vpolytope x zonotope
        VP_1 = VPolytope(V = np.array([1., 1.]))
        VP_2 = VPolytope(V = np.array([0., 1.]))
        VP_3 = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., -1.]]))
        VP_4 = VPolytope(V = np.array([[1., 0.], [0., 0.], [0., 1.]]))
        Z = Zonotope(c = [1., 1.])

        result_1 = VP_1.convex_hull(VP_2)
        result_2 = VP_3.convex_hull(VP_4)
        result_3 = VP_3.convex_hull(Z)

        true_result_1 = VPolytope(V = np.array([[1., 1.], [0., 1.]]))
        true_result_2 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        true_result_3 = VPolytope(V = np.array([[1., 1.], [-1., 0.], [0., -1.]]))

        assert result_1 == true_result_1
        assert result_2 == true_result_2
        assert result_3 == true_result_3

    def test_hpolyedron(self):
        ''' Test for conversion to HPolyhedron '''
        # cases:
        # - single vertex
        VP_1 = VPolytope(V = np.array([1., 0.]))

        with self.assertRaises(NotImplementedError):
            VP_1.hpolyhedron()
    
    def test_intersects(self):
        ''' Test for intersection check '''
        # cases:
        # - vpolytope x vector
        # - vpolytope x vpolytope (self)
        # - vpolytope x vpolytope
        # - vpolytope x single vertex
        # - vpolytope x interval
        # - vpolytope x hpolyhedron #todo

        VP_1 = VPolytope(V = np.array([[-1., 0.], [1., 1.], [0., -1.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([0.25, 0.25]))
        I = Interval(lb = [-1., 0.], ub = [0., 1.])

        assert VP_1.intersects(np.array([0., 0.]))
        assert VP_1.intersects(np.array([1., 1.]))
        assert VP_1.intersects(VP_1)
        assert VP_1.intersects(VP_2)
        assert VP_2.intersects(VP_1)
        assert VP_1.intersects(VP_3)
        assert VP_3.intersects(VP_1)
        assert VP_1.intersects(I)
    
    def test_interval(self):
        ''' Test for interval conversion '''
        # cases:
        # - single vertex
        # - vpolytope that is an interval
        # - vpolytope that is not an interval
        # todo: mode = 'exact' where not possible
        VP_1 = VPolytope(V = np.array([1., 1.]))
        VP_2 = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., 2.], [-1., 2.]]))
        VP_3 = VPolytope(V = np.array([[-1., 0.], [0., -1.], [2., 1.]]))

        result_1 = Interval(**VP_1.interval())
        result_2 = Interval(**VP_2.interval(mode = 'outer')) # should work with mode='exact'
        result_3 = Interval(**VP_3.interval(mode = 'outer'))

        I_1 = Interval(lb = [1., 1.], ub = [1., 1.])
        I_2 = Interval(lb = [-1., 0.], ub = [0., 2.])
        I_3 = Interval(lb = [-1., -1.], ub = [2., 1.])

        assert result_1 == I_1
        assert result_2 == I_2
        assert result_3 == I_3

        # unsupported conversions
        with self.assertRaises(NotImplementedError):
            VP_3.interval(mode = 'inner')
    
    def test_matmul(self):
        ''' Test for linear map '''
        # cases:
        # - single vertex
        # - multiple vertices, subspace
        VP_1 = VPolytope(V = np.array([1., 2.]))
        VP_2 = VPolytope(V = np.array([[1., 2.], [-1., 0.], [0., -1.]]))

        M1 = np.array([[2., 1.], [-1., 0.]])
        M2 = np.array([[1., -1.]])

        result_1 = VP_1.matmul(M1)
        result_2 = VP_2.matmul(M2)

        true_result_1 = VPolytope(V = np.array([4., -1.]))
        true_result_2 = VPolytope(V = np.array([[-1.], [-1.], [1.]]))

        assert result_1 == true_result_1
        assert result_2 == true_result_2
    
    def test_minkowski_sum(self):
        ''' Test for Minkowski sum '''
        # cases:
        # - single vertex + single vertex
        # - single vertex + multiple vertices
        # - multiple vertices + vector
        # - vpolytope + zonotope
        VP_1 = VPolytope(V = np.array([2., 1.]))
        VP_2 = VPolytope(V = np.array([-1., 4.]))
        VP_3 = VPolytope(V = np.array([[1., 0.], [-1., -1.], [-1., 1.]]))
        Z = Zonotope(c = np.array([2., 1.]))

        result_1 = VP_1.minkowski_sum(VP_2)
        result_2 = VP_2.minkowski_sum(VP_1)
        result_3 = VP_1.minkowski_sum(VP_3)
        result_4 = VP_3.minkowski_sum(np.array([2., 1.]))
        result_5 = VP_3.minkowski_sum(Z)

        VP_12 = VPolytope(V = np.array([1., 5.]))
        VP_13 = VPolytope(V = np.array([[3., 1.], [1., 0.], [1., 2.]]))

        assert result_1 == VP_12
        assert result_2 == VP_12
        assert result_3 == VP_13
        assert result_4 == VP_13
        assert result_5 == VP_13

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - vpolytope - vector
        # - vpolytope - vpolytope (vector)
        # - vpolytope - vpolytope
        VP_1 = VPolytope(V = np.array([[2., 0.], [-1., -1.], [-2., 1.]]))
        VP_2 = VPolytope(V = np.array([1., -1.]))
        VP_3 = VPolytope(V = np.array([[2., 0.], [-1., 1.]]))

        result_1 = VP_1.minkowski_difference(np.array([1., -1.]))
        result_2 = VP_1.minkowski_difference(VP_2)

        true_result_1 = VPolytope(V = np.array([[1., 1.], [-2., 0.], [-3., 2.]]))

        assert result_1 == true_result_1
        assert result_2 == true_result_1
        with self.assertRaises(NotImplementedError):
            VP_1.minkowski_difference(VP_3)

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[2., 3., 5., -1.]])
        V2 = np.array([[-1., 0., -1.], [0., -1., 0.], [0., 1., 2.], [1., -1., 1.], [1., 1., 1.], [2., 2., 0.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        VP_1_proj = VPolytope(V = np.array([[2., -1.]]))
        VP_2_proj = VPolytope(V = np.array([[0., -1.], [-1., 0.], [1., 2.], [-1., 1.], [1., 1.], [2., 0.]]))

        assert VP_1_proj == VP_1.project(axis = (0, 3))
        assert VP_2_proj == VP_2.project(axis = (1, 2))

    def test_represents(self):
        ''' Test for representation check '''
        # cases:
        # - single vertex
        # - multiple vertices
        VP_1 = VPolytope(V = np.array([2., 1.]))
        VP_2 = VPolytope(V = np.array([[-1., 0.], [2., 0.], [2., 1.], [-1., 1.]]))

        assert VP_1.represents(set_class = 'VPolytope')
        assert VP_1.represents(set_class = 'HPolyhedron')
        assert VP_1.represents(set_class = 'Interval')
        assert VP_1.represents(set_class = 'Zonotope')
        assert VP_2.represents(set_class = 'HPolyhedron')
        with self.assertRaises(NotImplementedError):
            VP_2.represents(set_class = 'Interval')

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - single vertex
        # - multiple vertices incl. redundancies
        V1 = np.array([[2., 3.]])
        V2 = np.array([[-1., 0.], [0., -1.], [0., 0.], [0., 1.], [1., -1.], [1., 1.], [2., 0.], [2., 2.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = V2)

        value1, vector1 = VP_1.support_function(np.array([1., 0.]))
        value2, vector2 = VP_2.support_function(np.array([1., 1.]))

        assert value1 == 2 and np.array_equal(vector1, np.array([2., 3.]))
        assert value2 == 4 and np.array_equal(vector2, np.array([2., 2.]))

    def test_vertices(self):
        ''' Test for vertex enumeration '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP_2 = VPolytope(V = V2)

        assert comparison.compare_matrices(VP_1.vertices(), V1)
        assert comparison.compare_matrices(VP_2.vertices(), V2)

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP_2 = VPolytope(V = V2)

        assert VP_1.volume() == 0
        with self.assertRaises(NotImplementedError):
            VP_2.volume()

    def test_vpolytope(self):
        ''' Test for overloaded conversion to vpolytope '''
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP_2 = VPolytope(V = V2)

        result1 = VP_1.vpolytope()
        result2 = VP_2.vpolytope()

        assert comparison.compare_matrices(result1['V'], V1)
        assert comparison.compare_matrices(result2['V'], V2)

    def test_zonotope(self):
        ''' Test for zonotope conversion '''
        # cases:
        # - single vertex
        # - multiple vertices
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP_2 = VPolytope(V = V2)

        result_1 = Zonotope(**VP_1.zonotope())
        result_2 = Zonotope(**VP_2.zonotope(mode = 'outer'))

        true_result_1 = Zonotope(c = V1.flatten())

        assert result_1 == true_result_1
        with self.assertRaises(NotImplementedError):
            assert result_2.contains(VP_2)

        # unsupported conversions
        with self.assertRaises(NotImplementedError):
            VP_2.zonotope(mode = 'exact')

if __name__ == '__main__':
    unittest.main()
