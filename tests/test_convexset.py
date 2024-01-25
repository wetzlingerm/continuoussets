import unittest
import numpy as np
from continuoussets.convexsets.convexset import ConvexSet
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.interval import Interval

class TestConvexSet(unittest.TestCase):

    def test_validate_input_arguments(self):
        ''' Test for validation flag '''
        # cases:
        # - interval

        # validation flag is on
        ConvexSet.validate_input_arguments(True)
        with self.assertRaises(ValueError):
            # ill-defined interval instantiation
            I = Interval(lb = np.array([1., 0.]), ub = np.array([2., 5., 2.]))

        # disable input argument validation 
        ConvexSet.validate_input_arguments(False)
        # not, ill-defined interval gets instantiated
        I = Interval(lb = np.array([1., 0.]), ub = np.array([2., 5., 2.]))

        # set validation flag again to True
        ConvexSet.validate_input_arguments(True)

        # check exceptions
        with self.assertRaises(TypeError):
            ConvexSet.validate_input_arguments("something")

    def test_ne(self):
        ''' Test not equal '''
        # cases:
        # - interval x interval

        # init intervals
        I1 = Interval(lb = np.array([-1., 0.]), ub = np.array([2., 1.]))
        I2 = Interval(lb = np.array([-1., 1.]), ub = np.array([2., 1.]))

        # check results
        assert (I1 == I1) != (I1 != I1)
        assert (I1 == I2) != (I1 != I2)

    def test_check_mode(self):
        ''' Test for mode check '''
        # cases:
        # - interval/cartesian_product

        # init interval
        I = Interval(lb = np.array([1., 0.]), ub = np.array([2., 1.]))
        
        # check exception
        with self.assertRaises(ValueError):
            I.cartesian_product(I, mode = 'something')

    def test_check_set_class(self):
        ''' Test for set class check '''
        # cases:
        # - interval/represents

        # init interval
        I = Interval(lb = np.array([1., 0.]), ub = np.array([2., 1.]))
        
        # check exception
        with self.assertRaises(ValueError):
            I.represents(set_class = 'something')

    def test_check_other_operand(self):
        ''' Test for operand check '''
        # cases:
        # - interval/convex_hull

        # init intervals
        I1 = Interval(lb = np.array([1., 0.]), ub = np.array([2., 1.]))
        I2 = Interval(lb = np.array([1., 0., 1.]), ub = np.array([2., 1., 1.]))
        M = np.array([[2., 1.], [3., -1.]])
        v = np.array([3., 2., 1.])

        # check exceptions
        with self.assertRaises(TypeError):
            # other operand is not a ConvexSet or np.ndarray
            I1.convex_hull("something")
        with self.assertRaises(AttributeError):
            # other operand is an np.ndarray and represents a matrix
            I1.convex_hull(M)
        with self.assertRaises(AttributeError):
            # other operand is an np.ndarray vector and has a different dimension
            I1.convex_hull(v)
        with self.assertRaises(AttributeError):
            # other operand is a ConvexSet and has a different dimension
            I1.convex_hull(I2)

    def test_check_subspace(self):
        ''' Test for subspace check '''
        # cases:
        # - zonotope/project

        # init zonotope
        Z = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0., -1.], [2., 1., 1.]]))

        # check exceptions
        with self.assertRaises(TypeError):
            # subspace is not a list or tuple
            Z.project(axis="something")
        with self.assertRaises(ValueError):
            # dimension too large
            Z.project(axis=(1, 2))
        with self.assertRaises(ValueError):
            # dimension too small
            Z.project(axis=(-1, 0))
        with self.assertRaises(ValueError):
            # dimension non-integer
            Z.project(axis=(0, 1.5))
        with self.assertRaises(ValueError):
            # repeated dimensions
            Z.project(axis=(0, 0))

    def test_check_matrix(self):
        ''' Test for left-multiplication of matrix '''
        # cases:
        # - zonotope/matmul

        # init zonotope
        Z = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0., -1.], [2., 1., 1.]]))

        # check exceptions
        with self.assertRaises(TypeError):
            # matrix is not a np.ndarray
            Z.matmul("something")
        with self.assertRaises(AttributeError):
            # matrix dimension does not fit zonotope
            M = np.array([[2., 1., 0.],[-1., 1., 1.]])
            Z.matmul(M)

if __name__ == '__main__':
    unittest.main()