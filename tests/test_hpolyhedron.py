import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions, auxiliary
from continuoussets.convexsets.hpolyhedron import HPolyhedron
from continuoussets.convexsets.vpolytope import VPolytope
from continuoussets.convexsets.interval import Interval
from continuoussets.convexsets.zonotope import Zonotope

class TestHPolyhedron(unittest.TestCase):

    def test_init(self):
        ''' Test for object instantation '''
        # cases:
        # - A, b: int, int
        # - A, b: float, float
        # - A, b: list, int
        # - A, b: list, float
        # - A, b: 2D list, float
        # - A, b: numpy

        # init HPolyhedron objects
        HP_int = HPolyhedron(A = 1, b = 1)
        HP_float = HPolyhedron(A = 1., b = 1.)
        HP_list_int = HPolyhedron(A = [1, 0, 0], b = 1)
        HP_list_float = HPolyhedron(A = [1., 0., 0.], b = 1.)
        HP_list2D_float = HPolyhedron(A = [[1., 0., 0.], [-1., 0., 0.]], b = [2., -1.])
        HP_numpy_float = HPolyhedron(A = np.array([[1., 0., 0.], [-1., 0., 0.]]), b = np.array([2., -1.]))

        # check results
        assert HP_int.dimension == 1
        assert HP_float.dimension == 1
        assert HP_list_int.dimension == 3
        assert HP_list_float.dimension == 3
        assert HP_list2D_float.dimension == 3
        assert HP_list2D_float.number_constraints() == 2
        assert HP_numpy_float.dimension == 3
        assert HP_numpy_float.number_constraints() == 2

        # check exceptions
        with self.assertRaises(ValueError):
            # no input arguments provided
            HPolyhedron()
        with self.assertRaises(ValueError):
            # not enough input arguments provided
            HPolyhedron(A = 1)
        with self.assertRaises(TypeError):
            # constraints are of wrong type
            HPolyhedron(A = "constraint", b = 1)
        with self.assertRaises(TypeError):
            # constraints are of wrong type
            HPolyhedron(A = 1, b = "constraint")
        with self.assertRaises(ValueError):
            # constraint matrix is >2D
            HPolyhedron(A = np.array([[[1., 0.], [1., 0.]], [[-1., 0.], [0., 1.]]]),
                        b = np.array([1., 0.]))
        with self.assertRaises(ValueError):
            # constraint offset is 2D
            HPolyhedron(A = np.array([[1., 0.]]), b = np.array([[1.], [1.]]))
        with self.assertRaises(ValueError):
            # number of constraints do not match
            HPolyhedron(A = np.array([[1., 0.], [0., 1.]]), b = np.array([2., 1., 1.]))

    def test_repr(self):
        ''' Test for display on the command window '''
        # cases:
        # - single constraint
        # - multiple constraints
        HP_single = HPolyhedron(A = np.array([1., 0.]), b = np.array([1.]))
        HP_multiple = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([1., 2., 1.]))

        # check if commands run through (check output manually)
        print(HP_single)
        print(HP_multiple)
        assert True

    def test_add(self):
        ''' Test for positive translation '''
        # cases:
        # - HPolyhedron x vector
        # - HPolyhedron x HPolyhedron (error)
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        vector = np.array([2., 1.])

        result1 = HP1 + vector

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([4., 2., 0.]))

        assert result1 == true_result1

        with self.assertRaises(exceptions.OtherFunctionError):
            HP1 + HP1

    def test_eq(self):
        ''' Test for set equality '''
        # cases:
        # - hpolyhedron x hpolyhedron
        # - hpolyhedron x vector
        # todo hpolyhedron x vpolytope
        # todo hpolyhedron x zonotope
        # todo hpolyhedron x interval
        HP1 = HPolyhedron(A = np.array([[-1., -1.], [1., 0.], [0., 1.]]),
                          b = np.array([-3., 2., 1.]))
        v1 = np.array([2., 1.])
        v2 = np.array([3., 1.])
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.], [0., -1.]]),
                          b = np.array([1., 1., 1., 5.]))
        HP4 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.], [0., -1.]]),
                          b = np.array([1., 1., 1., 0.5]))
        
        assert HP1 == v1
        assert not HP1 == v2
        assert HP2 == HP2
        assert HP2 == HP3
        assert not HP2 == HP4

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))

        result1 = -HP1

        true_result1 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.], [1., 1.]]), b = np.array([2., 1., 3.]))

        assert result1 == true_result1

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        result1 = +HP1

        #assert result1 == HP1
        assert True

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - HPolyhedron x vector
        # - HPolyhedron x HPolyhedron (error)
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        vector = np.array([2., 1.])

        result1 = HP1 - vector

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([0., 0., 6.]))

        assert result1 == true_result1

        with self.assertRaises(exceptions.OtherFunctionError):
            HP1 - HP1

    def test_basis_affine_hull(self):
        ''' Test for basis of affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate (1D in 2D)
        # - degenerate (1D in 2D, not containing origin)
        # todo degenerate unbounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1]]),
                          b = np.array([1., 0., 1., 0.]))
        HP3 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1]]),
                          b = np.array([7., -2., -5., 2.]))
        
        result1, r1 = HP1.basis_affine_hull()
        result2, r2 = HP2.basis_affine_hull()
        result3, r3 = HP3.basis_affine_hull()

        true_result2 = np.array([[-1./np.sqrt(2.), -1./np.sqrt(2.)], [-1./np.sqrt(2.), 1./np.sqrt(2.)]])

        assert np.array_equal(result1, np.eye(2))
        assert r1 == 2
        assert comparison.compare_matrices(result2, true_result2)
        assert r2 == 1
        assert comparison.compare_matrices(result3, true_result2)
        assert r3 == 1

    def test_boundary_point(self):
        ''' Test for boundary point computation '''
        # cases:
        # - not containing the origin
        # - bounded
        # - unbounded
        # - degenerate
        HP_noorigin = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([-0.5, 1., 1.]))
        HP_2D = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))
        HP_unbounded = HPolyhedron(A = np.array([[1., 0., 0.]]), b = np.array([2.]))
        HP_deg = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                             b = np.array([1., 0., 1., 0.]))

        result1 = HP_2D.boundary_point(np.array([1., -1.]))
        result2 = HP_2D.boundary_point(np.array([-1., 1.]))
        result3 = HP_2D.boundary_point(np.array([0., -1.]))
        result4 = HP_unbounded.boundary_point(np.array([1., 0., 0.]))

        true_result1 = np.array([1., -1.])
        true_result2 = np.array([-0.5, 0.5])
        true_result3 = np.array([0., -1.])
        true_result4 = np.array([2., 0., 0.])

        assert np.allclose(result1, true_result1)
        assert np.allclose(result2, true_result2)
        assert np.allclose(result3, true_result3)
        assert np.allclose(result4, true_result4)

        with self.assertRaises(NotImplementedError):
            HP_noorigin.boundary_point(np.array([1., 0.]))
        with self.assertRaises(exceptions.UnboundedSetError):
            HP_unbounded.boundary_point(np.array([-1., 0., 1.]))
        with self.assertRaises(NotImplementedError):
            HP_deg.boundary_point(np.array([1., 0.]))

    def test_bounded(self):
        ''' Test for boundedness check '''
        # cases:
        # - bounded
        # - unbounded
        # - empty
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.]]), b = np.ones(2))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([-1., -2., -1.]))
        HP4 = HPolyhedron(A = np.array([-1., -1.]), b = np.array([1.]))
        
        assert HP1.bounded()
        assert not HP2.bounded()
        assert HP3.bounded()
        assert not HP4.bounded()

    def test_cartesian_product(self):
        ''' Test for Cartesian product '''
        # cases:
        # - HPolytope x vector
        # - HPolytope x HPolytope
        # - HPolytope x Interval
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.]]), b = np.array([1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 2.]))
        I = Interval(lb = -2., ub = 3.)

        result1 = HP1.cartesian_product(np.array([2.]))
        result2 = HP1.cartesian_product(HP2)
        result3 = HP1.cartesian_product(I)

        true_result1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
                                   b = np.array([1., 1., 2., -2.]))
        true_result2 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
                                   b = np.array([1., 1., 3., 2.]))
        
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result2

    def test_center(self):
        ''' Test for center computation '''
        # cases:
        # - empty
        # - unbounded
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([-1., -2., -1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.]]), b = np.array([1., 2.]))
        HP3 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                          b = np.array([1., 1., 1., 1.]))
        
        with self.assertRaises(exceptions.EmptySetError):
            HP1.center()
        with self.assertRaises(exceptions.UnboundedSetError):
            HP2.center()
        assert np.array_equal(HP3.center(), np.zeros(2))

    def test_compact(self):
        ''' Test for minimal representation '''
        # cases:
        # - minimal
        # - single redundant constraint
        # - multiple redundant constraints (1D, 2D)
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([2., 1., 3.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [1., 1.], [0., 1.], [-1., -1.]]),
                          b = np.array([2., 10., 1., 3.]))
        HP3 = HPolyhedron(A = np.array([[1.], [1.], [1.], [1.]]),
                          b = np.array([1., 3., 2., 4.]))
        HP4 = HPolyhedron(A = np.array([[1., 0], [1., 1.], [-1., 1.], [-1., 0.], [0., -1.], [-1., -1.]]),
                          b = np.array([1., 4., 1., 1.5, 2.5, 1.]))
        
        result1 = HP1.compact()
        result2 = HP2.compact()
        result3 = HP3.compact()
        result4 = HP4.compact()

        assert result1.number_constraints() == 3
        assert result1 == HP1
        assert result2.number_constraints() == 3
        assert result2 == HP2
        assert result3.number_constraints() == 1
        assert result3 == HP3
        assert result4.number_constraints() == 3
        assert result4 == HP4

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - HPolyhedron x vector (inside)
        # - HPolyhedron x vector (boundary)
        # - HPolyhedron x HPolyhedron (self)
        # - HPolyhedron x HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([2., 1., 3.]))
        v_inside = np.array([0., 0.])
        v_boundary = np.array([2., 1.])
        HP2 = HPolyhedron(A = np.array([[1., 0.], [1., 1.]]),
                          b = np.array([5., 3.]))

        assert HP1.contains(v_inside)
        assert HP1.contains(v_boundary)
        assert HP1.contains(HP1)
        assert HP2.contains(HP1)
        assert not HP1.contains(HP2)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # cases:
        # - HPolyhedron x HPolyhedron
        # - HPolyhedron x vector
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([0., 0., 1.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.], [1., 1.]]), b = np.array([0., 0., 1.]))
        v = np.array([2., 1.])

        result1 = HP1.convex_hull(HP2, mode = 'outer')
        result2 = HP2.convex_hull(HP1, mode = 'outer')
        result3 = HP1.convex_hull(v, mode = 'outer')

        true_result1 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                                   b = np.array([1., 1., 1., 1.]))
        true_result3 = HPolyhedron(A = np.array([[1., -1.], [-1., 3.], [-1., -1.]]),
                                   b = np.array([1., 1., 1.]))
        
        assert result1.contains(HP1)
        assert result1.contains(HP2)
        assert result2.contains(HP1)
        assert result2.contains(HP2)
        assert result1.contains(true_result1)
        assert result2.contains(true_result1)
        assert result3.contains(true_result3)

        with self.assertRaises(NotImplementedError):
            HP1.convex_hull(HP2)

    def test_degenerate(self):
        ''' Test for degeneracy check '''
        # cases:
        # - empty
        # - non-degenerate
        # - degenerate
        # - unbounded non-degenerate
        # - unbounded degenerate
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                          b = np.array([0., 1., 0., 1.]))
        HP4 = HPolyhedron(A = np.array([[1., 0., 0.]]), b = np.array([2.]))
        HP5 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [-1., -1., 0.]]), b = np.zeros(3))

        assert HP1.degenerate()
        assert not HP2.degenerate()
        assert HP3.degenerate()
        assert not HP4.degenerate()
        assert HP5.degenerate()

    def test_empty(self):
        ''' Test for emptiness check '''
        # cases:
        # - 1D: empty, bounded, unbounded
        # - 2D: empty, bounded, unbounded
        # - 3D: single constraint
        HP_1D_bounded = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., 3.]))
        HP_1D_empty = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., -3.]))
        HP_1D_unbounded = HPolyhedron(A = np.array([[1.], [2.]]), b = np.array([1., 5.]))
        HP_2D_empty = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                                  b = np.array([0., 0., -1.]))
        HP_2D_unbounded = HPolyhedron(A = np.array([1., 0.]), b = np.array([1.]))
        HP_2D_bounded = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                    b = np.array([1., 1., 1.]))
        HP_3D_unbounded = HPolyhedron(A = np.array([[1., 0., 0.]]), b = np.array([1.]))

        assert not HP_1D_bounded.empty()
        assert HP_1D_empty.empty()
        assert not HP_1D_unbounded.empty()
        assert HP_2D_empty.empty()
        assert not HP_2D_bounded.empty()
        assert not HP_2D_unbounded.empty()
        assert not HP_3D_unbounded.empty()

    def test_hpolyhedron(self):
        ''' Test for overloaded conversion '''
        # cases:
        # - hpolyhedron
        A = np.array([[1., 0.], [-1., 1.], [-1., -1.]])
        b = np.array([1., 2., 1.])
        HP = HPolyhedron(A = A, b = b)

        d = HP.hpolyhedron()
        assert np.array_equal(d['A'], A)
        assert np.array_equal(d['b'], b)

    def test_intersection(self):
        ''' Test for intersection '''
        # cases:
        # - hpolyhedron x vector (contained)
        # - hpolyhedron x vector (not contained)
        # - hpolyhedron x hpolyhedron
        # - hpolyhedron x interval
        # todo hpolyhedron x zonotope
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 1.], [1., -1.]]), b = np.array([1., 1.]))
        I = Interval(lb = np.array([-2., 0.]), ub = np.array([0., 1.]))

        result1 = HP1.intersection(np.array([0., 0.]))
        result2 = HP1.intersection(HP2)
        result3 = HP2.intersection(HP1)
        result4 = HP1.intersection(I)

        true_result2 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                                   b = np.array([1., 1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [-1., 1.]]),
                                   b = np.array([0., 0., 1.]))

        assert result1 == np.array([0., 0.])
        assert HP1.intersection(np.array([10., 5.])).empty()
        assert result2 == true_result2
        assert result2 == result3
        assert result4 == true_result4

    def test_intersects(self):
        ''' Test for intersection check '''
        # cases:
        # - hpolyhedron x vector
        # - hpolyhedron x hpolyhedron
        # - hpolyhedron x interval
        # - hpolyhedron x zonotope
        # - hpolyhedron x vpolytope
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.], [1., 1.], [1., -1.]]),
                          b = np.array([2., 0., 0.]))
        I1 = Interval(lb = np.array([-1., -4.]), ub = np.array([4., 1.]))
        I2 = Interval(lb = np.array([2., -4.]), ub = np.array([4., 1.]))
        Z1 = Zonotope(c = np.array([1., 1.]), G = np.array([[1., -1.], [0., 1.]]))
        Z2 = Zonotope(c = np.array([2.5, 1.]), G = np.array([[1., -1.], [0., 1.]]))
        VP1 = VPolytope(V = np.array([[-1., -1.], [0., 0.], [-2., 2.]]))
        VP2 = VPolytope(V = np.array([[1.5, -2.], [3., 0.], [1., 4.]]))

        assert HP1.intersects(np.array([0., 0.]))
        assert not HP1.intersects(np.array([10., 5.]))
        assert HP1.intersects(HP2)
        assert HP2.intersects(HP1)
        assert HP1.intersects(I1)
        assert not HP1.intersects(I2)
        assert HP1.intersects(Z1)
        assert not HP1.intersects(Z2)
        assert HP1.intersects(VP1)
        assert not HP1.intersects(VP2)

    def test_interval(self):
        ''' Test for conversion to Interval '''
        # cases:
        # - bounded
        # - empty
        # - unbounded
        # - 1D
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.]]), b = np.array([1., 0.]))
        HP4 = HPolyhedron(A = np.array([[1.], [-1.], [1.]]), b = np.array([4., -2., 7.]))

        result1 = Interval(**HP1.interval(mode = 'outer'))
        result4 = Interval(**HP4.interval())

        true_result1 = Interval(lb = np.array([-1., -2.]), ub = np.array([1., 2.]))
        true_result4 = Interval(lb = np.array([2.]), ub = np.array([4.]))

        assert result1 == true_result1
        assert result4 == true_result4

        with self.assertRaises(exceptions.ExactEvaluationImpossibleError):
            HP1.interval()
        with self.assertRaises(NotImplementedError):
            HP1.interval(mode = 'inner')
        with self.assertRaises(exceptions.EmptySetError):
            HP2.interval(mode = 'outer')
        with self.assertRaises(exceptions.UnboundedSetError):
            HP3.interval(mode = 'outer')

    def test_matmul(self):
        ''' Test for linear map '''
        # cases:
        # - hpolyhedron x identity matrix
        # - hpolyhedron x square invertible matrix
        # - hpolyhedron x injective matrix
        # - hpolyhedron x surjective matrix
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                          b = np.ones(4))
        M1 = np.eye(2)
        M2 = np.array([[2., 1.], [-1., -1.]])
        M3 = np.array([[1., -1.]])
        M4 = np.array([[1., 2., -1.], [0., -1., 1.]])
        M5 = np.array([[1., 0.], [1., 1.], [-1., 1.]])

        result1 = HP1.matmul(M1)
        result2 = HP1.matmul(M2)
        result3 = HP1.matmul(M3)
        result4 = HP2.matmul(M4)

        true_result2 = HPolyhedron(A = np.array([[1., 1.], [-2., -3.], [0., 1.]]), b = np.ones(3))
        true_result3 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 1.]))
        true_result4 = HPolyhedron(A = np.array([[-0.5, -0.5], [0.5, 1.], [-0.5, -1.], [0.5, 0.5]]),
                                   b = np.ones(4))

        assert result1 == HP1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

        with self.assertRaises(NotImplementedError):
            HP1.matmul(M5)

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - hpolyhedron - interval
        # - hpolyhedron - vector
        HP = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                         b = np.array([1., 1., 1.]))
        I = Interval(lb = np.array([-0.1, -0.2]), ub = np.array([0.2, 0.3]))
        v = np.array([2., -1.])

        result1 = HP.minkowski_difference(I)
        result2 = HP.minkowski_difference(v)

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([0.8, 0.6, 0.7]))
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([-1., 4., 2.]))
        
        assert result1 == true_result1
        assert result2 == true_result2

    def test_minkowski_sum(self):
        ''' Test for Minkowski sum '''
        # cases:
        # - hpolyhedron + vector
        # - hpolyhedron + hpolyhedron
        # - hpolyhedron + interval
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        v = np.array([2., -1.])
        I = Interval(lb = np.array([-1., 0.]), ub = np.array([3., 1.]))

        result1 = HP1.minkowski_sum(v)
        result2 = HP1.minkowski_sum(HP1)
        result3 = HP1.minkowski_sum(HP1, mode = 'outer')
        result4 = HP1.minkowski_sum(I)
        result5 = HP1.minkowski_sum(I, mode = 'outer')

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([3., -2., 0.]))
        true_result2 = HPolyhedron(A = np.array([[0.5, 0.], [-0.5, 0.5], [-0.5, -0.5]]),
                                   b = np.array([1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[0., 1.], [0., -1.], [1., 0.], [-1., 0.], [-1./np.sqrt(2), 1./np.sqrt(2)], [-1./np.sqrt(2), -1./np.sqrt(2)]]),
                                   b = np.array([3., 2., 4., 2., 2.121320343559643, np.sqrt(2)]))
        
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3.contains(true_result2)
        assert result4 == true_result4
        assert result5.contains(true_result4)

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - 2D -> 1D
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        result1 = HP1.project(axis = (0,))

        true_result1 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., 1.]))

        assert result1 == true_result1

    def test_project_affine_hull(self):
        ''' Test for projection onto the basis of the affine hull '''
        # cases:
        # - non-degenerate
        # - degenerate
        # - degenerate, different center
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                          b = np.array([1., 0., 1., 0.]))
        HP3 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                          b = np.array([5., -8., -3., 8.]))
        # ...HP2 shifted by [6, -2]
        
        result1, M_proj1, c1 = HP1.project_affine_hull()
        result2, M_proj2, c2 = HP2.project_affine_hull()
        result3, M_proj3, c3 = HP3.project_affine_hull()

        true_result2 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([0., np.sqrt(2.)]))

        assert result1 == HP1
        assert np.array_equal(c1, np.zeros(2))
        assert result2 == true_result2
        assert result3 == true_result2

    def test_reduce(self):
        ''' Test for reduction of the set representation size '''
        # cases:
        # - bounded, non-degenerate
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        with self.assertRaises(NotImplementedError):
            HP1.reduce(order = 2)

    def test_represents(self):
        ''' Test for representation equivalence '''
        # cases:
        # - bounded
        # - 1D
        # - interval-like
        # - interval-like, but unbounded
        # - zonotope-like
        # - zonotope-like, but unbounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([4., -3.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., 2.], [-1., 0.], [0., -4], [1., 1.]]),
                          b = np.array([3., 2., 5., 1., 30.]))
        HP4 = HPolyhedron(A = np.array([[1., 0.], [0., 2.], [0., -4], [1., 1.]]),
                          b = np.array([3., 2., 1., 30.]))
        HP5 = HPolyhedron(A = np.array([[1., 0.], [0., 2.], [0., -4], [1., 1.]]),
                          b = np.array([3., 2., 1., 0.]))
        A6, b6 = auxiliary.halfspace_representation_from_vector(np.array([2., 1.]))
        HP6 = HPolyhedron(A = A6, b = b6)
        HP7 = HPolyhedron(A = np.array([[-0.5, -0.5], [0., -1/3.], [1./np.sqrt(2.), 1./np.sqrt(2.)], [0., 1.]]),
                          b = np.array([1., 1., 0., -1.]))
        HP8 = HPolyhedron(A = np.array([[1., 0.], [-1., 0.]]), b = np.ones(2))
        HP9 = HPolyhedron(A = np.array([[1., 0.], [0., 1.]]), b = np.ones(2))
        HP10 = HPolyhedron(A = np.array([[1., 0.], [-1., 0.]]), b = np.zeros(2))
        
        assert HP1.represents('HPolyhedron')
        assert HP1.represents('VPolytope')
        assert not HP1.represents('Interval')
        assert not HP1.represents('Zonotope')
        assert HP2.represents('Interval')
        assert HP2.represents('Zonotope')
        assert HP3.represents('Interval')
        assert not HP4.represents('Interval')
        assert not HP5.represents('Interval')
        assert not HP5.represents('Point')
        assert HP6.represents('Point')
        assert HP7.represents('Zonotope')
        assert not HP7.represents('Interval')
        assert not HP8.represents('Zonotope')
        assert not HP9.represents('Point')
        assert not HP10.represents('Point')

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - bounded
        # - unbounded
        # - empty
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.]]), b = np.array([2., 1.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [-1., 0.]]), b = np.array([1., -2.]))
        
        (value1, vector1) = HP1.support_function(np.array([1., 1.]))
        (value2, vector2) = HP2.support_function(np.array([1., 1.]))
        (value3, vector3) = HP3.support_function(np.array([1., 0.]))

        assert value1 == 3.
        assert np.array_equal(vector1, np.array([2., 1.]))
        assert value2 == np.inf
        assert value3 == -np.inf

    def test_vertices(self):
        ''' Test for vertex enumeration '''
        # cases:
        # - bounded
        # - degenerate (1D in 2D)
        # - empty
        # - unbounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                          b = np.array([5., -8., -3., 8.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP4 = HPolyhedron(A = np.array([[1., 0.], [1., 1.], [0., -1.]]),
                          b = np.array([1., 1., 1.]))
        
        V1 = HP1.vertices()
        V2 = HP2.vertices()
        
        true_result1 = np.array([[-1., 0.], [1., -2.], [1., 2.]])
        true_result2 = np.array([[5.5, -2.5], [6.5, -1.5]])

        assert comparison.compare_matrices(V1, true_result1)
        assert comparison.compare_matrices(V2, true_result2)
        
        with self.assertRaises(exceptions.EmptySetError):
            HP3.vertices()
        with self.assertRaises(exceptions.UnboundedSetError):
            V4 = HP4.vertices()

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - empty
        # - unbounded
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.]]), b = np.array([3., 2.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([1., 1., 1.]))
        
        assert HP1.volume() == 0.
        assert HP2.volume() == np.inf
        with self.assertRaises(NotImplementedError):
            HP3.volume()

    def test_vpolytope(self):
        ''' Test for vpolytope conversion '''
        # cases:
        # - bounded
        # - 1D
        # - degenerate  # todo
        # - unbounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.], [1.]]),
                          b = np.array([4., -2., 7.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1., -1.]))
        HP4 = HPolyhedron(A = np.array([[1., 0., 0.]]),
                          b = np.array([1.]))
        
        VP1 = VPolytope(**HP1.vpolytope())
        VP2 = VPolytope(**HP2.vpolytope())
        # VP3 = VPolytope(**HP3.vpolytope())

        true_result1 = VPolytope(V = np.array([[-1., 0.], [1., -2.], [1., 2.]]))
        true_result2 = VPolytope(V = np.array([[2.], [4.]]))
        true_result3 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))

        assert VP1 == true_result1
        assert VP2 == true_result2
        # assert VP3 == true_result3

        with self.assertRaises(exceptions.UnboundedSetError):
            VPolytope(**HP4.vpolytope())

    def test_zonotope(self):
        ''' Test for zonotope conversion '''
        # cases:
        # - bounded
        # - 1D
        # - zonotope-like
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.], [1.]]), b = np.array([4., -2., 7.]))
        HP3 = HPolyhedron(A = np.array([[0., -1.], [-1., -1.], [0.25, -0.5], [0., 1./3.], [1./7., 1./7.], [-0.25, 0.5]]),
                          b = np.ones(6))
        
        result1 = Zonotope(**HP1.zonotope(mode = 'outer'))
        result2 = Zonotope(**HP2.zonotope())

        true_result2 = Zonotope(c = np.array([3.]), G = np.array([[1.]]))

        assert result1.contains(HP1)
        assert result2 == true_result2

        with self.assertRaises(exceptions.ExactEvaluationImpossibleError):
            HP1.zonotope()
        with self.assertRaises(NotImplementedError):
            HP3.zonotope()
        with self.assertRaises(NotImplementedError):
            HP1.zonotope(mode = 'inner')

if __name__ == '__main__':
    unittest.main()
