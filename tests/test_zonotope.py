import unittest
import numpy as np
import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.interval import Interval
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.hpolyhedron import HPolyhedron


class TestZonotope(unittest.TestCase):

    def test_init(self):
        ''' Test for object instantation '''
        # cases:
        # - center: int
        # - center: float
        # - center: list
        # - center: np.ndarray

        # init zonotopes
        Z = Zonotope(c = 1, G = None)
        Z_int = Zonotope(c = 1)
        Z_float = Zonotope(c = -1., G = 1.)
        Z_list_int_1D = Zonotope(c = 1, G = [1])
        Z_list_int = Zonotope(c = [0, 1], G = [[1, 0], [-1, 2]])
        Z_list_float = Zonotope(c = [0., 1.], G = [[1., 0.], [-1., 2.]])
        Z_np = Zonotope(c = np.array([0., 1.]), G = [[1., 0.], [-1., 2.]])

        # check results
        assert np.array_equal(Z_int.c, np.array([1.]))
        assert np.array_equal(Z_float.c, np.array([-1.]))
        assert np.array_equal(Z_float.G, np.array([[1.]]))
        assert np.array_equal(Z_list_int_1D.c, np.array([1.]))
        assert np.array_equal(Z_list_int_1D.G, np.array([[1.]]))
        assert np.array_equal(Z_list_int.c, np.array([0., 1.]))
        assert np.array_equal(Z_list_int.G, np.array([[1., 0.],[-1., 2.]]))
        assert np.array_equal(Z_list_float.c, np.array([0., 1.]))
        assert np.array_equal(Z_list_float.G, np.array([[1., 0.],[-1., 2.]]))
        assert np.array_equal(Z_np.c, np.array([0., 1.]))
        assert np.array_equal(Z_np.G, np.array([[1., 0.],[-1., 2.]]))

        # check exceptions
        with self.assertRaises(ValueError):
            # no input arguments provided
            Zonotope()
        with self.assertRaises(TypeError):
            # wrong type for center
            Zonotope(c = 'center')
        with self.assertRaises(ValueError):
            # no center provided
            Zonotope(G = np.array([[1., 0.], [-1., 1.]]))
        with self.assertRaises(ValueError):
            # center is >1D
            Zonotope(c = np.array([[1.],[2.]]))
        with self.assertRaises(ValueError):
            # generator matrix does not match center dimension
            Zonotope(c = np.array([2., 1.]), G = np.array([[1., 0., -1.]]))
        with self.assertRaises(TypeError):
            # wrong type for generator matrix
            Zonotope(c = 1, G = 'generators')

    def test_repr(self):
        ''' Test for display on command window '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # check if commands run through
        print(Z1)
        print(Z2)
        assert True

    def test_add(self):
        ''' Test for positive translation '''
        # cases:
        # - zonotope + vector
        # - zonotope + zonotope
        # - zonotope + interval (error)
        # - zonotope + vpolytope (error)
        # - zonotope + hpolyhedron (error)

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([-2., 0.])

        # compute translation
        result1 = Z1 + v

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

        # call minkowski_sum instead of __add__
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 + Z1
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 + Interval(lb = np.array([1., 0.]), ub = np.array([2., 4.]))
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 + VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 + HPolyhedron(A = np.array([[1., 0.]]), b = np.array([1.]))

    def test_radd(self):
        ''' Test for positive translation '''
        # cases:
        # - vector + zonotope

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([-2., 0.])

        # compute translation
        result1 = v + Z1

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - only center
        # - center and generators

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., 0.], [0., 2.], [-1., 1.], [2., -1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # unary minus
        result1 = -Z1
        result2 = -Z2

        # manual computation
        true_result1 = Zonotope(c = -center)
        true_result2 = Zonotope(c = -center, G = generators)

        # check result
        assert result1 == true_result1
        assert result2 == true_result2

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - only center
        # - center and generators

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., 0.], [0., 2.], [-1., 1.], [2., -1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # unary plus
        result1 = +Z1
        result2 = +Z2

        # check result
        assert result1 == Z1
        assert result2 == Z2

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - zonotope - vector
        # - zonotope - zonotope
        # - zonotope - interval (error)
        # - zonotope - vpolytope (error)
        # - zonotope - hpolyhedron (error)

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([2., 0.])

        # compute translation
        result1 = Z1 - v

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

        # call minkowski_difference insetead of __sub__ with two IConvexSet objects
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 - Z1
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 - Interval(lb = np.array([0., 1.]), ub = np.array([2., 4.]))
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 - VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        with self.assertRaises(exceptions.OtherFunctionError):
            Z1 - HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

    def test_rsub(self):
        ''' Test for negative translation '''
        # cases:
        # - vector - zonotope

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([2., 0.])

        # compute translation
        result1 = v - Z1

        # manual computation
        true_result1 = Zonotope(c = np.array([1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

    def test_basis_affine_hull(self):
        ''' Test for computation of basis of the affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate
        Z1 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 1.], [-1., 0.], [2., 1.]]))
        Z2 = Zonotope(c = np.array([3., 2., -1.]),
                      G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.]]))

        result1, r1 = Z1.basis_affine_hull()
        result2, r2 = Z2.basis_affine_hull()

        # check whether multiplication with basis yields lower-dimensional rank of generator matrix
        assert np.all(np.all(np.isclose(np.matmul(result1.T, Z1.G.T), 0.), axis = 1) == np.full((2,), False))
        assert r1 == 2
        assert np.all(np.all(np.isclose(np.matmul(result2.T, Z2.G.T), 0.), axis = 1) == np.array([False, False, True]))
        assert r2 == 2

    def test_boundary_point(self):
        ''' Test for computation of boundary points '''
        # cases:
        # - full zonotope
        # - only center
        # - degenerate

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [-2., 1.], [2., 0.], [0., 1.]])
        generators_deg = np.array([[1., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        Z2 = Zonotope(c = center)
        Z3 = Zonotope(c = center, G = generators_deg)

        # compute boundary point
        direction = np.array([5., 3.])
        result1 = Z1.boundary_point(direction)
        result2 = Z2.boundary_point(direction)
        result3 = Z3.boundary_point(direction)

        # manual computation
        true_result1 = np.array([25/11., 15/11.]) + center

        # check results
        assert np.allclose(result1, true_result1)
        assert np.allclose(result2, center)
        assert np.allclose(result3, center)

    def test_bounded(self):
        ''' Test for boundedness '''
        # - only center
        # - center and generator matrix
        Z1 = Zonotope(c = np.array([1., 0., 1.]))
        Z2 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., -1.], [0., 2.]]))

        assert Z1.bounded()
        assert Z2.bounded()

    def test_center(self):
        ''' Test for computation of center '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # compute/read out center
        assert np.array_equal(center, Z1.center())
        assert np.array_equal(center, Z2.center())

    def test_compact(self):
        ''' Test for compact representation '''
        # cases:
        # - only center
        # - center and only all-zero generators
        # - center and some all-zero generators
        # - center and no all-zero generators
        # - center and aligned generators

        # init zonotopes
        center = np.array([1., 0.])
        generators_allzero = np.array([[0., 0.], [0., 0.], [0., 0.]])
        generators_somezero = np.array([[0., 0.], [1., 0.], [0., -1.]])
        generators_nozero = np.array([[1., 1.], [-2., 0.], [0., -1.]])
        generators_aligned = np.array([[1., -1.], [2., 0.], [-1., 1.], [0., 1.], [2., 1.], [3., 1.5], [1., 0.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators_allzero)
        Z3 = Zonotope(c = center, G = generators_somezero)
        Z4 = Zonotope(c = center, G = generators_nozero)
        Z5 = Zonotope(c = center, G = generators_aligned)

        # compact representation
        result1 = Z1.compact()
        result2 = Z2.compact()
        result3 = Z3.compact()
        result4 = Z4.compact()
        result5 = Z5.compact()

        # center always unchanged
        assert np.array_equal(Z1.c, result1.c)
        assert np.array_equal(Z2.c, result2.c)
        assert np.array_equal(Z3.c, result3.c)
        assert np.array_equal(Z4.c, result4.c)
        assert np.array_equal(Z5.c, result5.c)
        # all-zero generators are removed
        assert result1.G.size == 0
        assert result2.G.size == 0
        assert comparison.compare_matrices(result3.G, np.array([[1., 0.], [0., -1.]]), check_negation=True)
        assert comparison.compare_matrices(result4.G, Z4.G, check_negation=True)
        assert comparison.compare_matrices(result5.G, np.array([[2., -2.], [3., 0.], [0., 1.], [5., 2.5]]), check_negation=True)

    def test_degenerate(self):
        ''' Test for degeneracy '''
        # cases:
        # - only center
        # - degenerate
        # - non-degenerate
        Z1 = Zonotope(c = np.array([1., 0., 1.]))
        Z2 = Zonotope(c = np.array([1., 0., 1.]),
                      G = np.array([[1., 0., -1.], [-1., 0., 1.], [0., 1., 0.], [0., -1., 0.]]))
        Z3 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., -1.], [0., 2.]]))

        assert Z1.degenerate()
        assert Z2.degenerate()
        assert not Z3.degenerate()

    def test_empty(self):
        ''' Test for emptiness '''
        # cases:
        # - only center
        # - center and generator matrix
        Z1 = Zonotope(c = np.array([1., 0., 1.]))
        Z2 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., -1.], [0., 2.]]))

        assert not Z1.empty()
        assert not Z2.empty()

    def test_matmul(self):
        ''' Test for linear map '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [-2., 1.], [0., 3.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # compute linear map
        matrix = np.array([[1., -1.],[2., 0.]])
        matrix_subspace = np.array([[1., 2.]])
        result1 = Z1.matmul(matrix)
        result2 = Z1.matmul(matrix_subspace)
        result3 = Z2.matmul(matrix)
        result4 = Z2.matmul(matrix_subspace)

        # manual computation
        true_result1 = Zonotope(c = np.array([1., 2.]))
        true_result2 = Zonotope(c = np.array([1.]))
        true_result3 = Zonotope(c = np.array([1., 2.]), G = np.array([[2., 2.], [-3., -4.], [-3., 0.]]))
        true_result4 = Zonotope(c = np.array([1.]), G = np.array([[-1.], [0.], [6.]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_plot(self):
        ''' Test for plotting '''
        # cases:
        # - single point
        # - degenerate (line)
        # - full-dimensional

        # init zonotopes
        center = np.array([1., 0.])
        single_generator = np.array([[1., -1.]])
        generators = np.array([[1., 0.], [-1., 1.], [2., 1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = single_generator)
        Z3 = Zonotope(c = center, G = generators)

        # call plot (interactive mode to avoid blocking)
        with plt.ion():
            Z1.plot(axis = (0,1))
            plt.close()
            Z2.plot(axis = (0,1))
            plt.close()
            Z3.plot(axis = (0,1))
            plt.close()

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0., -1.])
        generators = np.array([[1., -1., 0.], [2., 0., -1.], [2., -1., 2.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # project
        subspace1 = (0, 2)
        subspace2 = (1, 2)
        result1 = Z1.project(axis = subspace1)
        result2 = Z2.project(axis = subspace2)

        # manual computation
        true_result1 = Zonotope(c = np.array([1., -1.]))
        true_result2 = Zonotope(c = np.array([0., -1.]), G = np.array([[-1., 0.], [0., -1.], [-1., 2.]]))
        
        assert result1 == true_result1
        assert result2 == true_result2

    def test_project_affine_hull(self):
        ''' Test for projection on affine hull '''
        # cases:
        # - non-degenerate
        # - 3D degenerate (2D in 3D)
        # - 3D degenerate (1D in 3D)
        Z1 = Zonotope(c = np.array([1., 2.]), G = np.array([[1., -1.], [1., 0.]]))
        Z2 = Zonotope(c = np.array([3., 2., -1.]),
                      G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.]]))
        Z3 = Zonotope(c = np.array([0., 0., 0.]),
                      G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.]]))
        Z4 = Zonotope(c = np.array([1., 0., 0.]), G = np.array([[-1., 2., 1.]]))

        result1, _, c1 = Z1.project_affine_hull()
        result2, _, c2 = Z2.project_affine_hull()
        result3, _, c3 = Z3.project_affine_hull()
        result4, _, c4 = Z4.project_affine_hull()

        true_result2 = Zonotope(c = np.array([0., 0.]),
                                G = np.array([[-2.121320343559642, 1.224744871391589],
                                              [-2.121320343559643, -1.224744871391589],
                                              [-4.242640687119286, 0.]]))

        true_result4 = Zonotope(c = np.array([0.]),
                                G = np.array([[2.449489742783179]]))

        assert result1 == Z1
        assert np.array_equal(c1, np.zeros(2))
        assert result2 == true_result2
        assert np.allclose(c2, Z2.center())
        assert result3 == true_result2
        assert np.allclose(c3, Z3.center())
        assert result4 == true_result4
        assert np.allclose(c4, Z4.center())

    def test_reduce(self):
        ''' Test for zonotope order reduction '''
        # cases:
        # - only center
        # - order too large for reduction
        # - reduction to order 1
        # - reduction to larger order

        # init zonotopes
        center = np.array([1., 0.])
        Z1 = Zonotope(c = center)
        generators = np.array([[2., 3.], [0., -1.], [2., -1.], [3., 2.], [-4., 0.], [1., 2.]])
        Z2 = Zonotope(c = center, G = generators)
        single_generator = np.array([[0., 1.]])
        Z3 = Zonotope(c = center, G = single_generator)

        # reduce
        result1 = Z1.reduce(order = 1)
        result2 = Z2.reduce(order = 5)
        result3 = Z2.reduce(order = 1)
        result4 = Z2.reduce(order = 2)
        result5 = Z3.reduce(order = 1)

        # manual computation
        true_result1 = Z1
        true_result2 = Z2
        true_result3 = Zonotope(c = center, G = np.array([[12., 0.], [0., 9.]]))
        true_result4 = Zonotope(c = center, G = np.array([[2., 3.], [3., 2.], [7., 0.], [0., 4.]]))
        true_result5 = Z3

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

        # check exceptions
        with self.assertRaises(ValueError):
            # order has to be at least 1
            Z1.reduce(order = 0.5)

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - only center
        # - center and all-zero generators
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators_zero = np.array([[0., 0.]])
        generators = np.array([[1., 1.], [-1., 1.], [0., 3.], [2., -1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators_zero)
        Z3 = Zonotope(c = center, G = generators)

        # evaluate support function
        (value1, vector1) = Z1.support_function(np.array([1., 0.]))
        (value2, vector2) = Z1.support_function(np.array([0., -1.]))
        (value3, vector3) = Z2.support_function(np.array([0., -1.]))
        (value4, vector4) = Z3.support_function(np.array([0., 1.]))
        (value5, vector5) = Z3.support_function(np.array([-2., 1.]))

        # check results
        assert value1 == 1 and np.array_equal(vector1, center)
        assert value2 == 0 and np.array_equal(vector2, center)
        assert value3 == 0 and np.array_equal(vector3, center)
        assert value4 == 6 and np.array_equal(vector4, np.array([-1., 6.]))
        assert value5 == 10 and np.array_equal(vector5, np.array([-3., 4.]))

    def test_vertices(self):
        ''' Test for vertex enumeration '''
        # cases:
        # - only center
        # - single generator
        # - generator matrix with full rank

        # init zonotopes
        center = np.array([1., 0.])
        generator = np.array([[-1., 2.]])
        generators = np.array([[1., 2.], [0., -1.], [-1., 1.], [2., 0.]])
        Z_onlycenter = Zonotope(c = center)
        Z_singlegenerator = Zonotope(c = center, G = generator)
        Z_fulldim = Zonotope(c = center, G = generators)

        # compute vertices
        result1 = Z_onlycenter.vertices()
        result2 = Z_singlegenerator.vertices()
        result3 = Z_fulldim.vertices()

        # manual computation
        true_result1 = center
        true_result2 = np.array([[0., 2.],[2., -2.]])
        true_result3 = np.array([[-1., -4.], [3., -4.], [5., 0.], [5., 2.], [3., 4.], [-1., 4.], [-3., 0.], [-3., -2.]])

        # check results
        assert comparison.compare_matrices(result1, true_result1)
        assert comparison.compare_matrices(result2, true_result2)
        assert comparison.compare_matrices(result3, true_result3)

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - only center
        # - degenerate generator matrix
        # - box generator matrix
        # - rotated non-degenerate generator matrix

        # init zonotopes
        center = np.array([1., 0.])
        generators_degenerate = np.array([[0., 1.], [0., 0.]])
        generators_box = np.array([[2., 0.], [0., 1.]])
        generators = np.array([[-3., 2.], [-2., 3.], [-1., 4.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators_degenerate)
        Z3 = Zonotope(c = center, G = generators_box)
        Z4 = Zonotope(c = center, G = generators)
        
        # compute volume
        result1 = Z1.volume()
        result2 = Z2.volume()
        result3 = Z3.volume()
        result4 = Z4.volume()

        # manual computation
        true_result1 = 0.
        true_result2 = 0.
        true_result3 = 8.
        true_result4 = 80.

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert np.isclose(result4, true_result4)

    def test_zonotope_norm(self):
        ''' Test for zonotope norm '''
        # cases:
        # - only center (origin)
        # - only center (not origin)
        # - full zonotope

        # init zonotopes
        center_origin = np.zeros(2)
        center_notorigin = np.array([1., 0.])
        generators = np.array([[1., -1.], [-2., 1.], [2., 0.], [0., 1.]])
        Z1 = Zonotope(c = center_notorigin)
        Z2 = Zonotope(c = center_origin, G = generators)
        Z3 = Zonotope(c = center_notorigin, G = generators)

        # compute zonotope norm
        result1 = Z1.zonotope_norm(np.array([0., 0.]))
        result2 = Z1.zonotope_norm(np.array([1., 1.]))
        result3 = Z2.zonotope_norm(np.array([5., 3.]))
        result4 = Z2.zonotope_norm(np.array([-5., 3.]))

        # manual computation
        true_result1 = 0.
        true_result2 = np.inf
        true_result3 = 2.2
        true_result4 = 1.

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # center needs to be at origin if there are generators
            Z3.zonotope_norm(np.array([1., 0.]))

    def test_array_ufunc(self):
        ''' Test for overloading of right-operations '''
        # cases:
        # - number x zonotope

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., 2.], [0., -1.], [-1., 1.]])
        Z = Zonotope(c = center, G = generators)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            np.array([1., 0.,]) * Z

if __name__ == '__main__':
    unittest.main()
    