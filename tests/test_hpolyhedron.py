import unittest
import numpy as np
#import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
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
        assert True

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

    def test_boundary_point(self):
        ''' Test for boundary point computation '''
        # cases:
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., 1.]]), b = np.ones(3))

        with self.assertRaises(NotImplementedError):
            HP1.boundary_point(np.array([1., 1.]))

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

        with self.assertRaises(NotImplementedError):
            HP1.cartesian_product(np.array([2., 1.]))

        result1 = HP1.cartesian_product(HP2)
        result2 = HP1.cartesian_product(I)

        true_result1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
                                   b = np.array([1., 1., 3., 2.]))
        
        assert result1 == true_result1
        assert result2 == true_result1

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
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([2., 1., 3.]))
        
        with self.assertRaises(NotImplementedError):
            HP1.compact()

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - HPolyhedron x vector
        # - HPolyhedron x HPolyhedron (self)
        # - HPolyhedron x HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                          b = np.array([2., 1., 3.]))
        v = np.array([0., 0.])
        HP2 = HPolyhedron(A = np.array([[1., 0.], [1., 1.]]),
                          b = np.array([5., 3.]))

        assert HP1.contains(v)
        assert HP1.contains(HP1)
        assert HP2.contains(HP1)
        assert not HP1.contains(HP2)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # cases:
        # - HPolyhedron x HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([0., 0., 1.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.], [1., 1.]]), b = np.array([0., 0., 1.]))

        result1 = HP1.convex_hull(HP2, mode = 'outer')
        result2 = HP2.convex_hull(HP1, mode = 'outer')

        true_result1 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                                   b = np.array([1., 1., 1., 1.]))
        
        assert result1.contains(HP1)
        assert result1.contains(HP2)
        assert result2.contains(HP1)
        assert result2.contains(HP2)
        assert result1.contains(true_result1)
        assert result2.contains(true_result1)

    def test_degenerate(self):
        ''' Test for degeneracy check '''
        # cases:
        # - empty
        # - non-degenerate
        # - degenerate
        # - unbounded non-degenerate
        # todo: unbounded degenerate
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP3 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                          b = np.array([0., 1., 0., 1.]))
        HP4 = HPolyhedron(A = np.array([1., 0., 0.]), b = np.array([2.]))

        assert HP1.degenerate()
        assert not HP2.degenerate()
        assert HP3.degenerate()
        with self.assertRaises(NotImplementedError):
            HP4.degenerate()

    def test_empty(self):
        ''' Test for emptiness check '''
        # cases:
        # - 1D: empty, bounded, unbounded
        # - 2D: empty, bounded, unbounded
        HP_1D_bounded = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., 3.]))
        HP_1D_empty = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., -3.]))
        HP_1D_unbounded = HPolyhedron(A = np.array([[1.], [2.]]), b = np.array([1., 5.]))
        HP_2D_empty = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                                  b = np.array([0., 0., -1.]))
        HP_2D_unbounded = HPolyhedron(A = np.array([1., 0.]), b = np.array([1.]))
        HP_2D_bounded = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                    b = np.array([1., 1., 1.]))

        assert not HP_1D_bounded.empty()
        assert HP_1D_empty.empty()
        assert not HP_1D_unbounded.empty()
        assert HP_2D_empty.empty()
        assert not HP_2D_bounded.empty()
        assert not HP_2D_unbounded.empty()

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

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - hpolyhedron - interval
        HP = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                         b = np.array([1., 1., 1.]))
        I = Interval(lb = np.array([-0.1, -0.2]), ub = np.array([0.2, 0.3]))

        result1 = HP.minkowski_difference(I)

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([0.8, 0.6, 0.7]))
        
        assert result1 == true_result1

    def test_represents(self):
        ''' Test for representation equivalence '''
        # cases:
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        
        assert HP1.represents(set_class = 'HPolyhedron')
        assert HP1.represents(set_class = 'VPolytope')
        with self.assertRaises(NotImplementedError):
            HP1.represents(set_class = 'Zonotope')

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - bounded
        # - unbounded
        # - empty
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.],[0., -1.]]), b = np.array([2., 1.]))
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
        # - empty
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [1., 1.], [-1., 0.]]),
                          b = np.array([-1., 0., -2., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        
        with self.assertRaises(exceptions.EmptySetError):
            HP1.vertices()
        with self.assertRaises(NotImplementedError):
            HP2.vertices()

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
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))

        with self.assertRaises(NotImplementedError):
            HP1.vpolytope()

    def test_zonotope(self):
        ''' Test for zonotope conversion '''
        # cases:
        # - bounded
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))

        with self.assertRaises(NotImplementedError):
            HP1.zonotope()

if __name__ == '__main__':
    unittest.main()
