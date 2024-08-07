import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions, auxiliary
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.hpolyhedron import HPolyhedron
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

    def test_basis_affine_hull(self):
        ''' Test for basis of affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate (2D in 3D)
        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))
        VP2 = VPolytope(V = np.array([[4., -3., -3.], [2., -7., -5], [0., -5., -1.],
                                      [4., 3., 3.], [2., 5., 7.], [0., 1., 5.]]))

        result1 = VP1.basis_affine_hull()
        result2 = VP2.basis_affine_hull()

        assert np.array_equal(result1, np.eye(2))
        # expression below checks if mapped vertices are equal in exactly one dimension
        assert 1 == np.nonzero(np.all(np.isclose(np.diff(np.matmul(result2.T, VP2.V.T), axis=1), 0.), axis=1))[0].size

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

    def test_hpolyhedron(self):
        ''' Test for conversion to HPolyhedron '''
        # cases:
        # - single vertex
        # - degenerate
        # - non-degenerate
        # - 1D
        V_singlevertex = np.array([1., 0.])
        VP_1 = VPolytope(V = V_singlevertex)
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-2., -2.]]))
        VP_4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        HP_1 = HPolyhedron(**VP_1.hpolyhedron())
        HP_2 = HPolyhedron(**VP_2.hpolyhedron())
        HP_3 = HPolyhedron(**VP_3.hpolyhedron())
        HP_4 = HPolyhedron(**VP_4.hpolyhedron())

        A1, b1 = auxiliary.halfspace_representation_from_vector(V_singlevertex)
        true_result1 = HPolyhedron(A = A1, b = b1)
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]]),
                                   b = np.array([1., 1., 1., -1.]))
        true_result3 = HPolyhedron(A = np.array([[1., -1.5], [1., 1.], [-1.5, 1.]]),
                                   b = np.array([1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 1.]))

        assert HP_1 == true_result1
        assert HP_2 == true_result2
        assert HP_3 == true_result3
        assert HP_4 == true_result4
    
    def test_intersects(self):
        ''' Test for intersection check '''
        # cases:
        # - vpolytope x vector
        # - vpolytope x vpolytope (self)
        # - vpolytope x vpolytope
        # - vpolytope x single vertex
        # - vpolytope x interval
        # - vpolytope x zonotope
        # - vpolytope x hpolyhedron
        VP_1 = VPolytope(V = np.array([[-1., 0.], [1., 1.], [0., -1.]]))
        VP_2 = VPolytope(V = np.array([[1., 0.], [0., 0.], [0., 1.]]))
        VP_3 = VPolytope(V = np.array([0.25, 0.25]))
        I_1 = Interval(lb = [-1., 0.], ub = [0., 1.])
        I_2 = Interval(lb = [0.75, -1.], ub = [1., 0.])
        Z_1 = Zonotope(c = np.array([1., -1.]), G = np.array([[1., -1.], [0.5, 0.]]))
        Z_2 = Zonotope(c = np.array([1., -1.]), G = np.array([[1., 1.], [0.5, 0.]]))
        HP_1 = HPolyhedron(A = np.array([[0., 1.]]), b = np.array([-0.5]))
        HP_2 = HPolyhedron(A = np.array([[-1., 1.]]), b = np.array([-1.1]))

        assert VP_1.intersects(np.array([0., 0.]))
        assert VP_1.intersects(np.array([1., 1.]))
        assert VP_1.intersects(VP_1)
        assert VP_1.intersects(VP_2)
        assert VP_2.intersects(VP_1)
        assert VP_1.intersects(VP_3)
        assert VP_3.intersects(VP_1)
        assert VP_1.intersects(I_1)
        assert not VP_1.intersects(I_2)
        assert VP_1.intersects(Z_1)
        assert not VP_1.intersects(Z_2)
        assert VP_1.intersects(HP_1)
        assert not VP_1.intersects(HP_2)
    
    def test_interval(self):
        ''' Test for interval conversion '''
        # cases:
        # - single vertex
        # - vpolytope that is an interval
        # - vpolytope that is not an interval
        # - 1D
        VP_1 = VPolytope(V = np.array([1., 1.]))
        VP_2 = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., 2.], [-1., 2.]]))
        VP_3 = VPolytope(V = np.array([[-1., 0.], [0., -1.], [2., 1.]]))
        VP_4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        result_1 = Interval(**VP_1.interval())
        result_2 = Interval(**VP_2.interval())
        result_3 = Interval(**VP_3.interval(mode = 'outer'))
        result_4 = Interval(**VP_4.interval())

        I_1 = Interval(lb = [1., 1.], ub = [1., 1.])
        I_2 = Interval(lb = [-1., 0.], ub = [0., 2.])
        I_3 = Interval(lb = [-1., -1.], ub = [2., 1.])
        I_4 = Interval(lb = np.array([-1.]), ub = np.array([3.]))

        assert result_1 == I_1
        assert result_2 == I_2
        assert result_3 == I_3
        assert result_4 == I_4

        # unsupported conversions
        with self.assertRaises(NotImplementedError):
            VP_3.interval(mode = 'inner')
        with self.assertRaises(exceptions.ExactEvaluationImpossibleError):
            VP_3.interval(mode = 'exact')
    
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

    def test_represents(self):
        ''' Test for representation check '''
        # cases:
        # - single vertex
        # - multiple vertices (interval)
        # - multiple vertices (zonotope)
        # - uneven number of vertices (cannot be a zonotope or an interval)
        VP_1 = VPolytope(V = np.array([2., 1.]))
        VP_2 = VPolytope(V = np.array([[-1., 0.], [2., 0.], [2., 1.], [-1., 1.]]))
        VP_3 = VPolytope(V = np.array([[4., -1., 1.], [2., -3., 1.], [6., 3., -1.], [4., 1., -1.],
                                       [-2., -3., 3.], [2., 3., 1.], [0., 1., 1.], [2., -3., 3.],
                                       [0., -5., 3.], [4., 1., 1.], [-2., -3., 5.], [-4., -5., 5.],
                                       [0., 1., 3.], [-2., -1., 3.]]))
        VP_4 = VPolytope(V = np.array([[2., 1.], [0., 2.], [-1., -2.]]))

        assert VP_1.represents('Point')
        assert VP_1.represents('Interval')
        assert VP_1.represents('Zonotope')
        assert VP_1.represents('VPolytope')
        assert VP_1.represents('HPolyhedron')

        assert not VP_2.represents('Point')
        assert VP_2.represents('Interval')
        assert VP_2.represents('Zonotope')
        assert VP_2.represents('VPolytope')
        assert VP_2.represents('HPolyhedron')

        assert not VP_3.represents('Point')
        assert not VP_3.represents('Interval')
        assert VP_3.represents('Zonotope')
        assert VP_3.represents('VPolytope')
        assert VP_3.represents('HPolyhedron')

        assert not VP_4.represents('Interval')
        assert not VP_4.represents('Zonotope')

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
        # - multiple vertices (not a zonotope)
        # - multiple vertices (is a zonotope)
        # - 1D
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        VP_2 = VPolytope(V = np.array([[2., 1.], [-1., 2.], [0., -4.]]))
        VP_3 = VPolytope(V = np.array([[1., -6.], [3., -2.], [3., 0.], [1., 2.], [-1., -2.], [-1., -4.]]))
        VP_4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        result_1 = Zonotope(**VP_1.zonotope())
        result_2 = Zonotope(**VP_2.zonotope(mode = 'outer'))
        result_4 = Zonotope(**VP_4.zonotope())

        assert result_1 == VP_1
        assert result_2.contains(VP_2)
        assert result_4 == VP_4

        # unsupported conversions
        with self.assertRaises(exceptions.ExactEvaluationImpossibleError):
            VP_2.zonotope(mode = 'exact')
        with self.assertRaises(NotImplementedError):
            VP_2.zonotope(mode = 'inner')
        with self.assertRaises(NotImplementedError):
            VP_3.zonotope()

if __name__ == '__main__':
    unittest.main()
