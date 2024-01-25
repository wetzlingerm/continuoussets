import unittest
import numpy as np
import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.interval import Interval

class TestZonotope(unittest.TestCase):

    def test_init(self):
        ''' Test for object instantation '''
        # cases:
        # - center: int
        # - center: float
        # - center: list
        # - center: np.ndarray

        # init zonotopes
        Z_int = Zonotope(c = 1)
        Z_float = Zonotope(c = -1., G = 1.)
        Z_list_int = Zonotope(c = [0, 1], G = [[1, 0],[-1, 2]])
        Z_list_float = Zonotope(c = [0., 1.], G = [[1., 0.],[-1., 2.]])
        Z_np = Zonotope(c = np.array([0., 1.]), G = [[1., 0.],[-1., 2.]])

        # check results
        assert np.array_equal(Z_int.c, np.array([1.]))
        assert np.array_equal(Z_float.c, np.array([-1.]))
        assert np.array_equal(Z_float.G, np.array([[1.]]))
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
        with self.assertRaises(ValueError):
            # no center provided
            Zonotope(G = np.array([[1., 0.], [-1., 1.]]))
        with self.assertRaises(ValueError):
            # center is >1D
            Zonotope(c = np.array([[1.],[2.]]))
        with self.assertRaises(ValueError):
            # generator matrix does not match center dimension
            Zonotope(c = np.array([2., 1.]), G = np.array([[1.],[0.],[-1.]]))

    def test_repr(self):
        ''' Test for display on command window '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
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

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([-2., 0.])

        # compute translation
        result1 = Z1 + v

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(exceptions.OtherFunctionError):
            # call minkowski_sum instead of __add__
            Z1 + Z1

    def test_radd(self):
        ''' Test for positive translation '''
        # cases:
        # - vector + zonotope

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([-2., 0.])

        # compute translation
        result1 = v + Z1

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

    def test_eq(self):
        ''' Test for set equality '''
        # cases:
        # - only center
        # - center and all-zero generators
        # - center and generators
        # - center and -1*generators
        # - center and aligned generators
        # - zonotope x interval

        # init zonotopes
        center = np.array([1., 0.])
        center_3D = np.array([1., 0., 1.])
        generators_allzero = np.array([[0., 0., 0.], [0., 0., 0.]])
        generators = np.array([[1., 2., -1.],[2., 0., 1.]])
        generators_reordered = np.array([[2., 1., -1.],[0., 2., 1.]])
        generators_neg = np.array([[2., 1., 1.],[0., 2., -1.]])
        genreators_aligned1 = np.array([[1., 2., -1., 0., 2., 3., 1.],[-1., 0., 1., 1., 1., 1.5, 0.]])
        genreators_aligned2 = np.array([[-3., 0., 0., 2., 4., -1.],[0., 0.5, -0.5, -2., 2., -0.5]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators_allzero)
        Z3 = Zonotope(c = center, G = generators)
        Z4 = Zonotope(c = center, G = generators_reordered)
        Z5 = Zonotope(c = center, G = generators_neg)
        Z6 = Zonotope(c = center_3D)
        Z7 = Zonotope(c = center + np.array([1., 0.]))
        Z8 = Zonotope(c = center, G = genreators_aligned1)
        Z9 = Zonotope(c = center, G = genreators_aligned2)
        I = Interval(lb = center, ub = center)

        # check set equality
        assert Z1 == center
        assert Z1 == Z2
        assert Z3 == Z3
        assert Z3 == Z4
        assert Z3 == Z5
        assert not Z1 == Z6
        assert not Z1 == Z7
        assert Z8 == Z9
        assert Z1 == I

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - zonotope

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., 0., -1., 2.],[0., 2., 1., -1.]])
        Z = Zonotope(c = center, G = generators)

        # unary minus
        result1 = -Z

        # manual computation
        true_result1 = Zonotope(c = -center, G = generators)

        # check result
        assert result1 == true_result1

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - zonotope

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., 0., -1., 2.],[0., 2., 1., -1.]])
        Z = Zonotope(c = center, G = generators)

        # unary plus
        result1 = +Z

        # check result
        assert result1 == Z

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - zonotope - vector

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([2., 0.])

        # compute translation
        result1 = Z1 - v

        # manual computation
        true_result1 = Zonotope(c = np.array([-1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(exceptions.OtherFunctionError):
            # call minkowski_difference insetead of __sub__ with two ConvexSet objects
            Z1 - Z1

    def test_rsub(self):
        ''' Test for negative translation '''
        # cases:
        # - zonotope + vector

        # init zonotope and vector
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        v = np.array([2., 0.])

        # compute translation
        result1 = v - Z1

        # manual computation
        true_result1 = Zonotope(c = np.array([1., 0.]), G = generators)

        # check results
        assert result1 == true_result1

    def test_boundary_point(self):
        ''' Test for computation of boundary points '''
        # cases:
        # - full zonotope

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., -2., 2., 0.],[-1., 1., 0., 1.]])
        Z = Zonotope(c = center, G = generators)

        # compute boundary point
        direction = np.array([5., 3.])
        result1 = Z.boundary_point(direction)

        # manual computation
        true_result1 = np.array([25/11., 15/11.]) + center

        # check results
        assert np.all(np.isclose(result1, true_result1))

    def test_cartesian_product(self):
        ''' Test for Cartesian product '''
        # cases:
        # - zonotope with only center x zonotope with only center
        # - zonotope with only center x vector
        # - zonotope with only center x zonotope
        # - zonotope x vector
        # - zonotope x zonotope with only center
        # - zonotope x zonotope
        # - zonotope x interval

        # init zonotopes
        center1 = np.array([-1., 0.])
        center2 = np.array([3., 1.])
        generators1 = np.array([[2.],[1.]])
        generators2 = np.array([[2., 1., -1., 0.],[0., 1., -1., 4.]])
        Z1_onlycenter = Zonotope(c = center1)
        Z2_onlycenter = Zonotope(c = center2)
        Z3 = Zonotope(c = center1, G = generators1)
        Z4 = Zonotope(c = center2, G = generators2)
        I = Interval(lb = np.array([3., 1.]), ub = np.array([3., 1.]))

        # compute Cartesian product
        result1 = Z1_onlycenter.cartesian_product(Z2_onlycenter)
        result2 = Z1_onlycenter.cartesian_product(center2)
        result3 = Z1_onlycenter.cartesian_product(Z4)
        result4 = Z3.cartesian_product(center2)
        result5 = Z3.cartesian_product(Z2_onlycenter)
        result6 = Z3.cartesian_product(Z4)
        result7 = Z1_onlycenter.cartesian_product(I)
        
        # manual computation
        centers_stacked = np.array([-1., 0., 3., 1.])
        zero_generators2 = np.array([[0., 0., 0., 0.],[0., 0., 0., 0.],[2., 1., -1., 0.],[0., 1., -1., 4.]])
        generators1_zero = np.array([[2.],[1.],[0.],[0.]])
        generators1_generators2 = np.array([[2., 0., 0., 0., 0.],[1., 0., 0., 0., 0.],[0., 2., 1., -1., 0.],[0., 0., 1., -1., 4.]])
        true_result1 = Zonotope(c = centers_stacked)
        true_result2 = true_result1
        true_result3 = Zonotope(c = centers_stacked, G = zero_generators2)
        true_result4 = Zonotope(c = centers_stacked, G = generators1_zero)
        true_result5 = Zonotope(c = centers_stacked, G = generators1_zero)
        true_result6 = Zonotope(c = centers_stacked, G = generators1_generators2)
        true_result7 = true_result1

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7

    def test_center(self):
        ''' Test for computation of center '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., 2., 2.], [-1., 0., -1.]])
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
        generators_allzero = np.array([[0., 0., 0.], [0., 0., 0.]])
        generators_somezero = np.array([[0., 1., 0.], [0., 0., -1.]])
        generators_nozero = np.array([[1., -2., 0.], [1., 0., -1.]])
        generators_aligned = np.array([[1., 2., -1., 0., 2., 3., 1.],[-1., 0., 1., 1., 1., 1.5, 0.]])
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
        assert result1.G is None
        assert result2.G is None
        assert comparison.compare_matrices(result3.G, np.array([[1., 0.], [0., -1.]]), check_negation=True)
        assert comparison.compare_matrices(result4.G, Z4.G, check_negation=True)
        assert comparison.compare_matrices(result5.G, np.array([[2., 3., 0., 5.],[-2., 0., 1., 2.5]]), check_negation=True)

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - only center x point (inside)
        # - only center x point (outside)
        # - full-dimensional x point (inside)
        # - full-dimensional x point (boundary)
        # - full-dimensional x point (outside)
        # - zonotope x zonotope (only center)

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -2., 2., 0.],[-1., 1., 0., 1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # check containment
        assert Z1.contains(center)
        assert not Z1.contains(np.array([2., 0.]))
        assert Z2.contains(center)
        assert Z2.contains(np.array([3., 1.]))
        assert Z2.contains(np.array([4., 1.]))
        assert Z2.contains(Z1)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            Z2.contains(Z2)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # cases:
        # - only center x only center
        # - only center x center and generators
        # - center and generators x only center
        # - center and generators x center and generators
        # - zonotope x interval

        # init zonotopes
        center1 = np.array([1., 0.])
        center2 = np.array([-1., 2.])
        generators1 = np.array([[2., 3., 1.],[-1., 2., 0.]])
        generators2 = np.array([[-1., 2., 0., 1., 3.],[4., 0., 1., -1., 2.]])
        Z1 = Zonotope(c = center1)
        Z2 = Zonotope(c = center2)
        Z3 = Zonotope(c = center1, G = generators1)
        Z4 = Zonotope(c = center2, G = generators2)
        I = Interval(lb = center2, ub = center2)

        # compute linear combination
        result1 = Z1.convex_hull(Z2)
        result2 = Z2.convex_hull(Z1)
        result3 = Z3.convex_hull(Z2)
        result4 = Z2.convex_hull(Z3)
        result5 = Z3.convex_hull(Z4)
        result6 = Z4.convex_hull(Z3)
        result7 = Z1.convex_hull(I)

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 1.]), G = np.array([[1.],[-1.]]))
        true_result2 = true_result1
        true_result3 = Zonotope(c = np.array([0., 1.]), G = np.array([[1., 2., 3., 1.],[-1., -1., 2., 0.]]))
        true_result4 = Zonotope(c = np.array([0., 1.]), G = np.array([[-1., 2., 3., 1.],[1., -1., 2., 0.]]))
        true_result5 = Zonotope(c = np.array([0., 1.]),
                                G = np.array([[0.5, 2.5, 0.5, -1., -1.5, -0.5, -0.5, 1., 3.],\
                                              [1.5, 1., 0.5, 1., 2.5, -1., 0.5, -1., 2.]]))
        true_result6 = true_result5
        true_result7 = true_result1

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7

        # check exceptions
        with self.assertRaises(NotImplementedError):
            Z1.convex_hull(Z2, mode='inner')

    def test_intersects(self):
        ''' Test for intersection check '''
        # cases:
        # - zonotope x point (inside)
        # - zonotope x point (boundary)
        # - zonotope x point (outside)
        # - zonotope x interval (intersection)
        # - zonotope x interval (no intersection)
        # - zonotope x zonotope (intersection)
        # - zonotope x zonotope (no intersection)
        # - zonotope x zonotope (no generators)
        # - zonotope (no generators) x zonotope

        # init zonotopes and intervals
        center1 = np.array([1., 0.])
        generators1 = np.array([[1., -1., 2., 0.],[-1., 2., 1., 1.]])
        Z1 = Zonotope(c = center1, G = generators1)
        lower1 = np.array([2., 2.])
        upper1 = np.array([5., 3.])
        I1 = Interval(lb = lower1, ub = upper1)
        lower2 = np.array([-3., -5.])
        upper2 = np.array([-2., -2.])
        I2 = Interval(lb = lower2, ub = upper2)
        center1 = np.array([1., 0.])
        generators1 = np.array([[1., -1., 2., 0.],[-1., 2., 1., 1.]])
        Z2 = Z1 + np.array([3., -2.])
        Z3 = Z1 + np.array([5., 5.])
        Z4 = Zonotope(c = center1)

        # check results
        assert Z1.intersects(center1)
        assert Z1.intersects(np.array([3., 3.]))
        assert not Z1.intersects(np.array([4., 2.]))
        assert Z1.intersects(I1)
        assert not Z1.intersects(I2)
        assert Z1.intersects(Z2)
        assert not Z1.intersects(Z3)
        assert Z1.intersects(Z4)
        assert Z4.intersects(Z1)

    def test_interval(self):
        ''' Test for conversion from zonotope to interval '''
        # cases:
        # - only center
        # - center and generators (box)
        # - center and generators (not a box)

        # init zonotopes
        c = np.array([1., 0.])
        G_axisaligned = np.array([[1., 0., 2., 0.], [0., -1., 0., 0.]])
        G_notaxisaligned = np.array([[1., 0.], [-1., 1.]])
        Z1 = Zonotope(c = c)
        Z2 = Zonotope(c = c, G = G_axisaligned)
        Z3 = Zonotope(c = c, G = G_notaxisaligned)

        # convert to intervals (dictionary for constructor)
        result1 = Interval(**Z1.interval())
        result2 = Interval(**Z1.interval(mode='inner'))
        result3 = Interval(**Z1.interval(mode='exact'))
        result4 = Interval(**Z2.interval())
        result5 = Interval(**Z3.interval())

        # manual computation
        true_result1 = Interval(lb = c, ub = c)
        true_result2 = true_result1
        true_result3 = true_result1
        true_result4 = Interval(lb = np.array([-2., -1.]), ub = np.array([4., 1.]))
        true_result5 = Interval(lb = np.array([0., -2.]), ub = np.array([2., 2.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # inner approximation not supported in general case
            Z3.interval(mode='inner')
        with self.assertRaises(NotImplementedError):
            # exact conversion not supported in general case
            Z3.interval(mode='exact')

    def test_matmul(self):
        ''' Test for linear map '''
        # cases:
        # - only center
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -2., 0.],[-1., 1., 3.]])
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
        true_result3 = Zonotope(c = np.array([1., 2.]), G = np.array([[2., -3., -3.],[2., -4., 0.]]))
        true_result4 = Zonotope(c = np.array([1.]), G = np.array([[-1., 0., 6.]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_minkowski_sum(self):
        ''' Test for Minkowski sum '''
        # cases:
        # - zonotope + zonotope
        # - zonotope + vector
        # - zonotope + interval

        # init zonotopes
        center1 = np.array([1., 0.])
        generators1 = np.array([[1., 2., 2.], [-1., 0., -1.]])
        center2 = np.array([-1., 1.])
        generators2 = np.array([[0., -1., 3.], [1., -1., 0.]])
        Z1 = Zonotope(c = center1, G = generators1)
        Z2 = Zonotope(c = center2, G = generators2)
        v = np.array([-2., 0.])
        Z3 = Zonotope(c = v)
        I = Interval(lb = v, ub = v)

        # compute Minkowski sums
        result1 = Z1.minkowski_sum(Z2)
        result2 = Z1.minkowski_sum(v)
        result3 = Z1.minkowski_sum(I)
        result4 = Z3.minkowski_sum(Z1)

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 1.]),\
                                G = np.array([[1., 2., 2., 0., -1., 3.], [-1., 0., -1., 1., -1., 0.]]))
        true_result2 = Zonotope(c = np.array([-1., 0.]), G = generators1)
        true_result3 = true_result2
        true_result4 = true_result2

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - zonotope x vector
        
        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[0., 1., 2., -1.],[1., 2., 1., 0.]])
        Z = Zonotope(c = center, G = generators)

        # compute Minkowski difference
        result1 = Z.minkowski_difference(center)

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 0.,]), G = generators)

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # Minkowski difference between zonotopes not implemented
            Z.minkowski_difference(Z)

    def test_plot(self):
        ''' Test for plotting '''
        # cases:
        # - single point
        # - degenerate (line)
        # - full-dimensional

        # init zonotopes
        center = np.array([1., 0.])
        single_generator = np.array([[1.], [-1.]])
        generators = np.array([[1., -1., 2.],[0., 1., 1.]])
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
        generators = np.array([[1., 2., 2.], [-1., 0., -1.], [0., -1., 2.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)

        # project
        subspace1 = (0, 2)
        subspace2 = (1, 2)
        result1 = Z1.project(axis = subspace1)
        result2 = Z2.project(axis = subspace2)

        # manual computation
        true_result1 = Zonotope(c = np.array([1., -1.]))
        true_result2 = Zonotope(c = np.array([0., -1.]), G = np.array([[-1., 0., -1.],[0., -1., 2.]]))
        
        assert result1 == true_result1
        assert result2 == true_result2

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
        generators = np.array([[2., 0., 2., 3., -4., 1.],[3., -1., -1., 2., 0., 2.]])
        Z2 = Zonotope(c = center, G = generators)
        single_generator = np.array([[0.],[1.]])
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
        true_result3 = Zonotope(c = center, G = np.array([[12., 0.],[0., 9.]]))
        true_result4 = Zonotope(c = center, G = np.array([[2., 3., 7., 0.],[3., 2., 0., 4.]]))
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

    def test_represents(self):
        ''' Test for representation check '''
        # cases:
        # - only center
        # - center and generators (box)
        # - center and generators (not a box)

        # init zonotopes
        c = np.array([1., 0.])
        G_axisaligned = np.array([[1., 0., 2., 0.], [0., -1., 0., 0.]])
        G_notaxisaligned = np.array([[1., 0.], [-1., 1.]])
        Z1 = Zonotope(c = c)
        Z2 = Zonotope(c = c, G = G_axisaligned)
        Z3 = Zonotope(c = c, G = G_notaxisaligned)

        # check representation
        assert Z1.represents('Interval')
        assert Z2.represents('Interval')
        assert not Z3.represents('Interval')
        assert Z1.represents('Zonotope')

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - only center
        # - center and all-zero generators
        # - center and generators

        # init zonotopes
        center = np.array([1., 0.])
        generators_zero = np.array([[0.],[0.]])
        generators = np.array([[1., -1., 0., 2.],[1., 1., 3., -1.]])
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
        generator = np.array([[-1.], [2.]])
        generators = np.array([[1., 0., -1., 2.],[2., -1., 1., 0.]])
        Z_onlycenter = Zonotope(c = center)
        Z_singlegenerator = Zonotope(c = center, G = generator)
        Z_fulldim = Zonotope(c = center, G = generators)

        # compute vertices
        result1 = Z_onlycenter.vertices()
        result2 = Z_singlegenerator.vertices()
        result3 = Z_fulldim.vertices()

        # manual computation
        true_result1 = np.reshape(center, (2,1))
        true_result2 = np.array([[0., 2.],[2., -2.]])
        true_result3 = np.array([[-1., 3., 5., 5., 3., -1., -3., -3.],[-4., -4., 0., 2., 4., 4., 0., -2.]])

        # check results
        assert comparison.compare_matrices(result1, true_result1)
        assert comparison.compare_matrices(result2, true_result2)
        assert comparison.compare_matrices(result3, true_result3)

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - only center

        # init zonotopes
        center = np.array([1., 0.])
        generators_degenerate = np.array([[0., 1.],[0., 0.]])
        generators_box = np.array([[2., 0.],[0., 1.]])
        generators = np.array([[-3., -2., -1.],[2., 3., 4.]])
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

    def test_zonotope(self):
        ''' Test for conversion to zonotope '''
        # cases:
        # - zonotope

        # init zonotope
        center = np.array([1., 2.])
        generators = np.array([[0., -1.],[3., 4.]])
        Z = Zonotope(c = center, G = generators)

        # convert to zonotope
        result1 = Zonotope(**Z.zonotope())

        # manual computation
        true_result1 = Z

        # check result
        assert result1 == true_result1

    def test_zonotope_norm(self):
        ''' Test for zonotope norm '''
        # cases:
        # - only center (origin)
        # - only center (not origin)
        # - full zonotope

        # init zonotopes
        center_origin = np.zeros(2)
        center_notorigin = np.array([1., 0.])
        generators = np.array([[1., -2., 2., 0.],[-1., 1., 0., 1.]])
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
        generators = np.array([[1., 0., -1.],[2., -1., 1.]])
        Z = Zonotope(c = center, G = generators)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            np.array([1., 0.,]) * Z

if __name__ == '__main__':
    unittest.main()