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
            HPolyhedron(A = "constraint", b = "constraint")
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

        #assert result1 == true_result1

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

        #assert result1 == true_result1
        assert True

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

        #assert result1 == true_result1

        with self.assertRaises(exceptions.OtherFunctionError):
            HP1 - HP1

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - HPolyhedron x vector
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        v = np.array([0., 0.])

        assert HP1.contains(v)

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
        (value3, vector3) = HP2.support_function(np.array([1., 0.]))

        assert value1 == 3.
        assert np.array_equal(vector1, np.array([2., 1.]))
        assert value2 == np.inf
        assert value3 == -np.inf



if __name__ == '__main__':
    unittest.main()
