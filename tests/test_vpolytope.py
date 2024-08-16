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

    def test_basis_affine_hull(self):
        ''' Test for basis of affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate (2D in 3D)
        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))
        VP2 = VPolytope(V = np.array([[4., -3., -3.], [2., -7., -5], [0., -5., -1.],
                                      [4., 3., 3.], [2., 5., 7.], [0., 1., 5.]]))

        result1, r1 = VP1.basis_affine_hull()
        result2, r2 = VP2.basis_affine_hull()

        # expression below checks if mapped vertices are equal in exactly one dimension
        assert 0 == np.nonzero(np.all(np.isclose(np.diff(np.matmul(result1.T, VP1.V.T), axis=1), 0.), axis=1))[0].size
        assert r1 == 2
        assert 1 == np.nonzero(np.all(np.isclose(np.diff(np.matmul(result2.T, VP2.V.T), axis=1), 0.), axis=1))[0].size
        assert r2 == 2

    def test_boundary_point(self):
        ''' Test for boundary point computation '''
        # cases:
        # - single vertex
        # - 2D
        # - origin not contained
        VP_singlevertex = VPolytope(V = np.array([[4., -2.]]))
        VP_2D = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -2.]]))
        VP_far = VPolytope(V = np.array([[5., 3.], [3., 4.], [4., 1.]]))
        
        result1 = VP_2D.boundary_point(np.array([0., -1.]))
        result2 = VP_2D.boundary_point(np.array([1., 1.]))
        result3 = VP_2D.boundary_point(np.array([-1., -1.]))

        true_result1 = np.array([0., -1.])
        true_result2 = np.array([0.5, 0.5])
        true_result3 = np.array([-0.5, -0.5])

        assert np.allclose(result1, true_result1)
        assert np.allclose(result2, true_result2)
        assert np.allclose(result3, true_result3)

        # vpolytope degenerate / does not contain the origin
        with self.assertRaises(NotImplementedError):
            VP_singlevertex.boundary_point(np.array([-1., 0.]))
        with self.assertRaises(NotImplementedError):
            VP_far.boundary_point(np.array([1., 0.]))

    def test_bounded(self):
        ''' Test for boundedness '''
        # cases:
        # - single vertex
        # - degenerate
        # - non-degenerate
        VP_1 = VPolytope(V = np.array([[1., 0.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))

        assert VP_1.bounded()
        assert VP_2.bounded()
        assert VP_3.bounded()

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
    
    def test_degenerate(self):
        ''' Test for degeneracy '''
        # cases:
        # - single vertex
        # - degenerate
        # - non-degenerate
        VP_1 = VPolytope(V = np.array([[1., 0.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))

        assert VP_1.degenerate()
        assert VP_2.degenerate()
        assert not VP_3.degenerate()

    def test_empty(self):
        ''' Test for emptiness '''
        # cases:
        # - single vertex
        # - degenerate
        # - non-degenerate
        VP_1 = VPolytope(V = np.array([[1., 0.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))

        assert not VP_1.empty()
        assert not VP_2.empty()
        assert not VP_3.empty()
    
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

    def test_project_affine_hull(self):
        ''' Test for projection onto affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate
        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))
        VP2 = VPolytope(V = np.array([[4., -3., -3.], [2., -7., -5], [0., -5., -1.],
                                      [4., 3., 3.], [2., 5., 7.], [0., 1., 5.]]))
        
        result1, _, c1 = VP1.project_affine_hull()
        result2, _, c2 = VP2.project_affine_hull()
        # note: different centers are possible, but center needs to match the projection

        true_result2 = VPolytope(V = np.array([[4.242640687119285, -2.449489742783178],
                                               [8.485281374238570, 0.],
                                               [4.242640687119286, 2.449489742783178],
                                               [-4.242640687119286, -2.449489742783178],
                                               [-8.485281374238570, 0.],
                                               [-4.242640687119286, 2.449489742783178]]))

        assert result1 == VP1
        assert np.array_equal(c1, np.zeros(2))
        assert result2 == true_result2
        assert np.allclose(c2, np.array([2., -1., 1.]))

    def test_reduce(self):
        ''' Test for reduction of set representation size '''
        # cases:
        # - non-degenerate
        VP = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -2.]]))
        
        with self.assertRaises(NotImplementedError):
            VP.reduce(order = 2)

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
        # - degenerate set
        # - non-degenerate set (error)
        VP_1 = VPolytope(V = np.array([[2., 3., -1.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., 2.]]))
        VP_3 = VPolytope(V = np.array([[2., 1.], [-1., 2.], [0., -4.]]))

        assert VP_1.volume() == 0
        assert VP_2.volume() == 0

        with self.assertRaises(NotImplementedError):
            VP_3.volume()

if __name__ == '__main__':
    unittest.main()
