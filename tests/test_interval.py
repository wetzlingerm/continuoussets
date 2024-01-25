import unittest
import numpy as np
import matplotlib.pyplot as plt
from continuoussets.utils import comparison, exceptions
from continuoussets.convexsets.zonotope import Zonotope
from continuoussets.convexsets.interval import Interval


class TestInterval(unittest.TestCase):

    def test_init(self):
        ''' Test for object instantation '''
        # cases:
        # - single int
        # - single float
        # - list of floats
        # - list of int
        # - numpy array of floats
        # - only lower bound
        # - only upper bound

        # init intervals
        I1 = Interval(lb = -1, ub = 2)
        I2 = Interval(lb = -1., ub = 2.)
        I3 = Interval(lb = [1, 2], ub = [2, 3])
        I4 = Interval(lb = [1., 2.], ub = [2., 3.])
        I5 = Interval(lb = np.array([1., 2.]), ub = np.array([2., 3.]))
        I6 = Interval(lb = 1)
        I7 = Interval(ub = 1.)

        # full instantiation
        true_I1 = Interval(lb = np.array([-1.]), ub = np.array([2.]))
        true_I2 = true_I1
        true_I3 = I5
        true_I4 = I5
        true_I5 = I5
        true_I6 = Interval(lb = np.array([1.]), ub = np.array([1.]))
        true_I7 = true_I6

        # check results
        assert I1 == true_I1
        assert I2 == true_I2
        assert I3 == true_I3
        assert I4 == true_I4
        assert I5 == true_I5
        assert I6 == true_I6
        assert I7 == true_I7

        # check exceptions
        with self.assertRaises(TypeError):
            # no arguments provided
            Interval()
        with self.assertRaises(ValueError):
            # lower bound is 2D array
            Interval(lb = np.array([[2., 1.],[-1., 2.]]), ub = np.array([4., 5.]))
        with self.assertRaises(ValueError):
            # upper bound is 2D array
            Interval(lb = np.array([-4., -5.]), ub = np.array([[2., 1.],[-1., 2.]]))
        with self.assertRaises(ValueError):
            # wrong dimensions
            Interval(lb = 1, ub = [2, 1])
        with self.assertRaises(ValueError):
            # lower bound exceeds upper bound
            Interval(lb = 5, ub = 2)

    def test_getitem(self):
        ''' Test for indexing '''
        # cases:
        # - 0 index
        # - positive index
        # - negative index
        # - connected indices
        # - disconnected indices

        # init interval
        lower = np.array([1., -3., -2., 0., 3., 2., 6.])
        upper = np.array([4., -2., -2., 4., 4., 8., 7.])
        I = Interval(lb = lower, ub = upper)

        # index the interval
        result1 = I[0]
        result2 = I[1]
        result3 = I[-1]
        result4 = I[1:4]
        result5 = I[0:7:2]

        # manual computation
        true_result1 = Interval(lb = np.array([1.]), ub = np.array([4.]))
        true_result2 = Interval(lb = np.array([-3.]), ub = np.array([-2.]))
        true_result3 = Interval(lb = np.array([6.]), ub = np.array([7.]))
        true_result4 = Interval(lb = np.array([-3., -2., 0.]), ub = np.array([-2., -2., 4.]))
        true_result5 = Interval(lb = np.array([1., -2., 3., 6.]), ub = np.array([4., -2., 4., 7.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

    def test_setitem(self):
        ''' Test for setting indices '''
        # cases:
        # - 0 index
        # - positive index
        # - negative index
        # - connected indices
        # - disconnected indices

        # init intervals
        lower = np.array([-2., -4., 2., 3., -1.])
        upper = np.array([3., -1., 4., 3., 7.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper)
        I3 = Interval(lb = lower, ub = upper)
        I4 = Interval(lb = lower, ub = upper)
        I5 = Interval(lb = lower, ub = upper)

        # set index to different value
        I0 = Interval(lb = np.array([0.]), ub = np.array([2.]))
        I1[0] = I0
        I2[1] = I0
        I3[-1] = I0
        I_subspace = Interval(lb = np.array([0., -10.]), ub = np.array([10., 0.]))
        I4[0:2] = I_subspace
        I5[0:4:2] = I_subspace

        # check results
        assert I1[0] == I0
        assert I2[1] == I0
        assert I3[-1] == I0
        assert I4[0:2] == I_subspace
        assert I5[0:4:2] == I_subspace

    def test_repr(self):
        ''' Test for display on command window '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # print on command window
        print(I1)
        print(I2)
        print(I3)
        assert True

    def test_add(self):
        ''' Test for Minkowski sum '''
        # cases:
        # interval x vector
        # interval (single point) x vector
        # interval (single point) x interval
        # interval x interval
        # interval x int
        # interval x float
        # interval x list

        # init intervals
        lower1 = np.array([-2., 1., 0.])
        upper1 = np.array([2., 1., 4.])
        lower2 = np.array([-4., 2., 0.])
        upper2 = np.array([-2., 5., 1.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([3., 1., -2.])
        I2 = Interval(lb = vector, ub = vector)
        I3 = Interval(lb = lower2, ub = upper2)
        scalar_int = 2
        scalar_float = -2.
        vector_list = [1., -2., 0.]

        # Minkowski sum
        result1 = I1 + vector
        result2 = I2 + vector
        result3 = I1 + I2
        result4 = I2 + I1
        result5 = I1 + I3
        result6 = I3 + I1
        result7 = I1 + scalar_int
        result8 = I1 + scalar_float
        result9 = I1 + vector_list

        # manual computation
        true_result1 = Interval(lb = np.array([1., 2., -2.]), ub = np.array([5., 2., 2.]))
        true_result2 = Interval(lb = np.array([6., 2., -4.]), ub = np.array([6., 2., -4.]))
        true_result3 = true_result1
        true_result4 = true_result1
        true_result5 = Interval(lb = np.array([-6., 3., 0.]), ub = np.array([0., 6., 5.]))
        true_result6 = true_result5
        true_result7 = Interval(lb = np.array([0., 3., 2.]), ub = np.array([4., 3., 6.]))
        true_result8 = Interval(lb = np.array([-4., -1., -2.]), ub = np.array([0., -1., 2.]))
        true_result9 = Interval(lb = np.array([-1., -1., 0.]), ub = np.array([3., -1., 4.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8 == true_result8
        assert result9 == true_result9

    def test_radd(self):
        ''' Test for Minkowski sum '''
        # cases:
        # - scalar int x interval
        # - scalar float x interval
        # - list x interval
        # - np.ndarray x interval

        # init intervals
        lower = np.array([-2., 1., 0.])
        upper = np.array([2., 1., 4.])
        I = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = -2.
        vector_list = [1., -2., 0.]
        vector_np = np.array(vector_list)

        # Minkowski sum
        result1 = scalar_int + I
        result2 = scalar_float + I
        result3 = vector_list + I
        result4 = vector_np + I

        # manual computation
        true_result1 = Interval(lb = np.array([0., 3., 2.]), ub = np.array([4., 3., 6.]))
        true_result2 = Interval(lb = np.array([-4., -1., -2.]), ub = np.array([0., -1., 2.]))
        true_result3 = Interval(lb = np.array([-1., -1., 0.]), ub = np.array([3., -1., 4.]))
        true_result4 = true_result3

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_eq(self):
        ''' Test for set equality '''
        # cases:
        # interval x interval (itself)
        # interval x degenerate interval
        # interval x np.ndarray
        # single-point interval x np.ndarray (False)
        # single-point interval x np.ndarray (True)
        # intervals of different dimension
        # interval x zonotope

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)
        lower_3D = np.array([-2., -1., 0.])
        upper_3D = np.array([3., 4., 0.])
        I4 = Interval(lb = lower_3D, ub = upper_3D)
        Z = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 2.5]]))

        assert I1 == I1
        assert not I1 == I2
        assert not I1 == lower
        assert not I3 == (lower + np.array([1., 0.]))
        assert I3 == lower
        assert not I1 == I4
        assert I1 == Z

    def test_mul(self):
        ''' Test for elementwise multiplication '''
        # cases:
        # - interval x scalar (int)
        # - interval x scalar (float)
        # - interval x list
        # - interval x np.ndarray
        # - interval x interval

        # init intervals
        lower = np.array([-2., 2., 1.])
        upper = np.array([0., 4., 1.])
        I = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = -2.
        vector = [1., -2., 0.]
        vector_np = np.array([1., -2., 0.])

        # compute elementwise multiplication
        result1 = I * scalar_int
        result2 = I * scalar_float
        result3 = I * vector
        result4 = I * vector_np
        result5 = I * I

        # manual computation
        true_result1 = Interval(lb = np.array([-4., 4., 2.]), ub = np.array([0., 8., 2.]))
        true_result2 = Interval(lb = np.array([0., -8., -2.]), ub = np.array([4., -4., -2.]))
        true_result3 = Interval(lb = np.array([-2., -8., 0.]), ub = np.array([0., -4., 0.]))
        true_result4 = true_result3
        true_result5 = Interval(lb = np.array([0., 4., 1.]), ub = np.array([4., 16., 1.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

    def test_rmul(self):
        ''' Test for elementwise multiplication '''
        # cases:
        # - scalar(int) x interval
        # - scalar (float) x interval
        # - list x interval
        # - np.ndarray x interval

        # init intervals
        lower = np.array([-2., 2., 1.])
        upper = np.array([0., 4., 1.])
        I = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = -2.
        vector = [1., -2., 0.]
        vector_np = np.array([1., -2., 0.])

        # compute elementwise multiplication
        result1 = scalar_int * I
        result2 = scalar_float * I
        result3 = vector * I
        result4 = vector_np * I

        # manual computation
        true_result1 = Interval(lb = np.array([-4., 4., 2.]), ub = np.array([0., 8., 2.]))
        true_result2 = Interval(lb = np.array([0., -8., -2.]), ub = np.array([4., -4., -2.]))
        true_result3 = Interval(lb = np.array([-2., -8., 0.]), ub = np.array([0., -4., 0.]))
        true_result4 = true_result3

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_neg(self):
        ''' Test for unary minus operator '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-3., -1., 0., 1., 4.])
        upper = np.array([-2., 0., 2., 6., 10.])
        I = Interval(lb = lower, ub = upper)

        # apply unary minus
        result = -I

        # manual computation
        true_result = Interval(lb = np.array([2., 0., -2., -6., -10.]), ub = np.array([3., 1., 0., -1., -4.]))

        # check result
        assert result == true_result

    def test_pos(self):
        ''' Test for unary minus operator '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-3., -1., 0., 1., 4.])
        upper = np.array([-2., 0., 2., 6., 10.])
        I = Interval(lb = lower, ub = upper)

        # apply unary minus
        result = +I

        # check result
        assert result == I

    def test_pow(self):
        ''' Test for elementwise exponentiation '''
        # cases:
        # - interval x int
        # - interval x float
        # - interval x list
        # - interval x np.ndarray

        # init interval
        lower = np.array([-4., -3., -2., 0., 1.])
        upper = np.array([-2., 0., 1., 3., 4.])
        I = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = 3.
        vector_list = [1., 2., 3., 2., 1.]
        vector_np = np.array(vector_list)
        scalar_zero = 0.
        scalar_one = 1.
        vector_noninteger = [1., 2., 3., 1.5, 2.5]
        vector_neg = [1., 2., 3., 2., -2.5]

        # compute exponentiation
        result1 = I ** scalar_int
        result2 = I ** scalar_float
        result3 = I ** vector_list
        result4 = I ** vector_np
        result5 = I ** scalar_zero
        result6 = I ** scalar_one
        result7 = I ** vector_noninteger
        result8 = I ** vector_neg

        # manual computation
        true_result1 = Interval(lb = np.array([4., 0., 1., 0., 1.]),
                                ub = np.array([16., 9., 4., 9., 16.]))
        true_result2 = Interval(lb = np.array([-64., -27., -8., 0., 1.]),
                                ub = np.array([-8., 0., 1., 27., 64.]))
        true_result3 = Interval(lb = np.array([-4., 0., -8., 0., 1.]),
                                ub = np.array([-2., 9., 1., 9., 4.]))
        true_result4 = true_result3
        true_result5 = Interval(lb = np.ones(5), ub = np.ones(5))
        true_result6 = I
        true_result7 = Interval(lb = np.array([-4., 0., -8., 0., 1.]),
                                ub = np.array([-2., 9., 1., 3 ** 1.5, 4. ** 2.5]))
        true_result8 = Interval(lb = np.array([-4., 0., -8., 0., 1./32.]),
                                ub = np.array([-2., 9., 1., 9, 1.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8 == true_result8

        # check exceptions
        with self.assertRaises(ValueError):
            # smaller than 0 values raised to non-integer power
            I = Interval(lb = np.array([-1.]), ub = np.array([1.]))
            I ** 1.5
        with self.assertRaises(ValueError):
            # interval containing 0 raised to negative power
            I = Interval(lb = np.array([-1.]), ub = np.array([1.]))
            I ** -2.

    def test_sub(self):
        ''' Test for Minkowski sum including *-1 for second operand '''
        # cases:
        # interval x vector
        # interval (single point) x vector
        # interval (single point) x interval
        # interval x interval
        # interval x int
        # interval x float
        # interval x list

        # init intervals
        lower1 = np.array([-2., 1., 0.])
        upper1 = np.array([2., 1., 4.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([3., 1., -2.])
        I2 = Interval(lb = vector, ub = vector)
        lower2 = np.array([-4., 2., 0.])
        upper2 = np.array([-2., 5., 1.])
        I3 = Interval(lb = lower2, ub = upper2)
        scalar_int = 2
        scalar_float = -2.
        vector_list = [1., -2., 0.]

        # Minkowski sum
        result1 = I1 - vector
        result2 = I2 - vector
        result3 = I1 - I2
        result4 = I2 - I1
        result5 = I1 - I3
        result6 = I1 - scalar_int
        result7 = I1 - scalar_float
        result8 = I1 - vector_list

        # manual computation
        true_result1 = Interval(lb = np.array([-5., 0., 2.]), ub = np.array([-1., 0., 6.]))
        true_result2 = Interval(lb = np.array([0., 0., 0.]), ub = np.array([0., 0., 0.]))
        true_result3 = true_result1
        true_result4 = Interval(lb = np.array([1., 0., -6.]), ub = np.array([5., 0., -2.]))
        true_result5 = Interval(lb = np.array([0., -4., -1.]), ub = np.array([6., -1., 4.]))
        true_result6 = Interval(lb = np.array([-4., -1., -2.]), ub = np.array([0., -1., 2.]))
        true_result7 = Interval(lb = np.array([0., 3., 2.]), ub = np.array([4., 3., 6.]))
        true_result8 = Interval(lb = np.array([-3., 3., 0.]), ub = np.array([1., 3., 4.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8 == true_result8

    def test_rsub(self):
        ''' Test for Minkowski sum including *(-1) for second operand '''
        # cases:
        # - scalar int - interval
        # - scalar float - interval
        # - list - interval
        # - np.ndarray - interval

        # init intervals
        scalar_int = 2
        scalar_float = -3.
        vector_list = [1., -2., 0., -1., 3]
        vector_np = np.array(vector_list)
        lower = np.array([-4., -1., -1., 0., 4.])
        upper = np.array([-2., 0., 2., 3., 6.])
        I = Interval(lb = lower, ub = upper)

        # compute Minkowski sum
        result1 = scalar_int - I
        result2 = scalar_float - I
        result3 = vector_list - I
        result4 = vector_np - I

        # manual computation
        true_result1 = Interval(lb = np.array([4., 2., 0., -1., -4.]), ub = np.array([6., 3., 3., 2., -2.]))
        true_result2 = Interval(lb = np.array([-1., -3., -5., -6., -9.]), ub = np.array([1., -2., -2., -3., -7.]))
        true_result3 = Interval(lb = np.array([3., -2., -2., -4., -3.]), ub = np.array([5., -1., 1., -1., -1.]))
        true_result4 = true_result3

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_truediv(self):
        ''' Test for elementwise division '''
        # cases:
        # - interval / int
        # - interval / float
        # - interval / list
        # - interval / np.ndarray
        # - interval / interval

        # init intervals
        lower = np.array([-4., -1., -1., 0., 4.])
        upper = np.array([-2., 0., 2., 3., 6.])
        I1 = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = -3.
        vector_list = [2., 1., -1., -4., 3.]
        vector_np = np.array(vector_list)
        lower2 = np.array([-2., -4., 1., 2., 3.])
        upper2 = np.array([-1., -2., 1., 3., 5.])
        I2 = Interval(lb = lower2, ub = upper2)

        # compute division
        result1 = I1 / scalar_int
        result2 = I1 / scalar_float
        result3 = I1 / vector_list
        result4 = I1 / vector_np
        result5 = I1 / I2

        # manual computation
        true_result1 = Interval(lb = np.array([-2., -0.5, -0.5, 0., 2.]), ub = np.array([-1., 0., 1., 1.5, 3.]))
        true_result2 = Interval(lb = np.array([2/3., 0., -2/3., -1., -2.]), ub = np.array([4/3., 1/3., 1/3., 0., -4/3.]))
        true_result3 = Interval(lb = np.array([-2., -1., -2., -0.75, 4/3.]), ub = np.array([-1., 0., 1., 0., 2.]))
        true_result4 = true_result3
        true_result5 = Interval(lb = np.array([1., 0., -1., 0., 0.8]), ub = np.array([4., 0.5, 2., 1.5, 2.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

        # check exceptions
        with self.assertRaises(ZeroDivisionError):
            # interval x interval: division by zero
            I1 = Interval(lb = -1., ub = 1.)
            I1 / I1
        with self.assertRaises(ZeroDivisionError):
            # interval x interval: division by zero
            I1 = Interval(lb = -1., ub = 1.)
            I1 / 0.


    def test_rtruediv(self):
        ''' Test for elementwise division '''
        # cases:
        # - int / interval
        # - float / interval
        # - list / interval
        # - np.ndarray / interval

        # init intervals
        lower = np.array([-4., 1., -2., 2.])
        upper = np.array([-2., 3., -1., 5.])
        I = Interval(lb = lower, ub = upper)
        scalar_int = 2
        scalar_float = -3.
        vector_list = [-2., 3., 0., -1.]
        vector_np = np.array(vector_list)

        # compute division
        result1 = scalar_int / I
        result2 = scalar_float / I
        result3 = vector_list / I
        result4 = vector_np / I

        # manual computation
        true_result1 = Interval(lb = np.array([-1., 2/3., -2., 0.4]), ub = np.array([-0.5, 2., -1., 1.]))
        true_result2 = Interval(lb = np.array([0.75, -3., 1.5, -1.5]), ub = np.array([1.5, -1., 3., -0.6]))
        true_result3 = Interval(lb = np.array([0.5, 1., 0., -0.5]), ub = np.array([1., 3., 0., -0.2]))
        true_result4 = true_result3

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_absolute_value(self):
        ''' Test for absolute value '''
        # cases:
        # - lower and upper >= 0
        # - lower and upper <= 0
        # - mixed bound >= 0 and <= 0
        # - single point

        # init intervals
        lower_larger_zero = np.array([3., 1., 5.])
        upper_larger_zero = np.array([6., 2., 7.])
        lower_less_zero = np.array([-2., -8., -4.])
        upper_less_zero = np.array([-1., -6., -3.])
        vector = np.array([1., -2., 0., 3.])
        I1 = Interval(lb = lower_larger_zero, ub = upper_larger_zero)
        I2 = Interval(lb = lower_less_zero, ub = upper_less_zero)
        I3 = Interval(lb = lower_larger_zero, ub = lower_larger_zero)
        I4 = Interval(lb = vector, ub = vector)

        # compute absolute value
        result1 = I1.absolute_value()
        result2 = I2.absolute_value()
        result3 = I3.absolute_value()
        result4 = I4.absolute_value()
        
        # manual computation
        true_result1 = I1
        true_result2 = Interval(lb = np.abs(upper_less_zero), ub = np.abs(lower_less_zero))
        true_result3 = I3
        true_result4 = Interval(lb = np.array([1., 2., 0., 3.]), ub = np.array([1., 2., 0., 3.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_arccos(self):
        ''' Test for arccosine '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-0.5, -np.sqrt(3)/2.])
        upper = np.array([1., 0.])
        I = Interval(lb = lower, ub = upper)

        # compute arccosine
        result1 = I.arccos()

        # manual computation
        true_result1 = Interval(lb = np.array([0., np.pi/2.]), ub = np.array([2.*np.pi/3., 5.*np.pi/6.]))

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # lower bound below -1
            I = Interval(lb = np.array([-1.5]), ub = np.array([0.5]))
            I.arccos()
        with self.assertRaises(exceptions.OutOfBoundsError):
            # upper bound above 1
            I = Interval(lb = np.array([-0.5]), ub = np.array([1.5]))
            I.arccos()

    def test_arcsin(self):
        ''' Test for arcsine '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-0.5, -np.sqrt(3)/2.])
        upper = np.array([1., 0.])
        I = Interval(lb = lower, ub = upper)

        # compute arccosine
        result1 = I.arcsin()

        # manual computation
        true_result1 = Interval(lb = np.array([-np.pi/6., -np.pi/3.]), ub = np.array([np.pi/2., 0.]))

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # lower bound below -1
            I = Interval(lb = np.array([-1.5]), ub = np.array([0.5]))
            I.arcsin()
        with self.assertRaises(exceptions.OutOfBoundsError):
            # upper bound above 1
            I = Interval(lb = np.array([-0.5]), ub = np.array([1.5]))
            I.arcsin()

    def test_arctan(self):
        ''' Test for arctangent '''
        # cases:
        # - interval

        # init interval
        lower = np.array([0., -np.sqrt(3)])
        upper = np.array([1., -np.sqrt(3)/3.])
        I = Interval(lb = lower, ub = upper)

        # compute arctangent
        result1 = I.arctan()

        # manual computation
        true_result1 = Interval(lb = np.array([0., -np.pi/3.]), ub = np.array([np.pi/4., -np.pi/6.]))

        # check results
        assert result1 == true_result1

    def test_boundary_point(self):
        ''' Test for computation of boundary points '''
        # cases:
        # - axis-aligned directions
        # - arbitrary direction

        # init intervals
        lower = np.array([-2., -4.])
        upper = np.array([3., 1.])
        I = Interval(lb = lower, ub = upper)

        # directions
        first_axis = np.array([1., 0.])
        second_axis = np.array([0., -1.])
        first_quadrant = np.array([1., 1.])
        second_quadrant = np.array([-2., 1.])
        third_quadrant = np.array([-1., -2.])
        fourth_quadrant = np.array([2., -1.])

        # compute boundary point
        result1 = I.boundary_point(first_axis)
        result2 = I.boundary_point(second_axis)
        result3 = I.boundary_point(first_quadrant)
        result4 = I.boundary_point(second_quadrant)
        result5 = I.boundary_point(third_quadrant)
        result6 = I.boundary_point(fourth_quadrant)

        # manual computation
        true_result1 = np.array([3., 0.])
        true_result2 = np.array([0., -4.])
        true_result3 = np.array([1., 1.])
        true_result4 = np.array([-2., 1.])
        true_result5 = np.array([-2., -4.])
        true_result6 = np.array([3., -1.5])

        # check results
        assert np.array_equal(result1, true_result1)
        assert np.array_equal(result2, true_result2)
        assert np.array_equal(result3, true_result3)
        assert np.array_equal(result4, true_result4)
        assert np.array_equal(result5, true_result5)
        assert np.array_equal(result6, true_result6)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # sets that do not contain the origin not supported
            I = Interval(lb = np.array([1., 2.]), ub = np.array([2., 4.]))
            I.boundary_point(np.array([1., 0.]))
        with self.assertRaises(NotImplementedError):
            # degenerate sets not supported
            I = Interval(lb = np.array([-1., 0.]), ub = np.array([1., 0.]))
            I.boundary_point(np.array([1., 0.]))

    def test_cartesian_product(self):
        ''' Test for Cartesian product '''
        # cases:
        # - interval x interval
        # - interval x np.ndarray
        # - interval x zonotope

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        I1 = Interval(lb = lower, ub = upper)
        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[-1., 0.],[0., 2.]])
        Z = Zonotope(c = center, G = generators)

        # compute Cartesian product
        result1 = I1.cartesian_product(I1)
        result2 = I1.cartesian_product(lower)
        result3 = I1.cartesian_product(Z)

        # manual computation
        true_result1 = Interval(lb = np.hstack((lower, lower)),\
                                ub = np.hstack((upper, upper)))
        true_result2 = Interval(lb = np.hstack((lower, lower)),\
                                ub = np.hstack((upper, lower)))
        true_result3 = Interval(lb = np.hstack((lower, np.array([0., -2.]))),\
                                ub = np.hstack((upper, np.array([2., 2.]))))
        
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

        with self.assertRaises(NotImplementedError):
            # 'exact' not supported for Interval x Zonotope
            I = Interval(lb = np.array([-1., 0.]), ub = np.array([1., 0.]))
            Z = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0., -1., 2.],[-2., 1., 1., 0.]]))
            I.cartesian_product(Z, mode = 'exact')

    def test_center(self):
        ''' Test for computation of center '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # compute center
        result1 = I1.center()
        result2 = I2.center()
        result3 = I3.center()
        
        # manual computation
        true_result1 = np.array([0.5, 1.5])
        true_result2 = np.array([-2., 1.5])
        true_result3 = lower

        # check results
        assert np.array_equal(result1, true_result1)
        assert np.array_equal(result2, true_result2)
        assert np.array_equal(result3, true_result3)

    def test_compact(self):
        ''' Test for compact representation '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-2., 0., 3.])
        upper = np.array([-1., 1., 5.])
        I = Interval(lb = lower, ub = upper)

        # compute compact representation
        result1 = I.compact()

        # manual computation
        true_result1 = I

        # check results
        assert result1 == true_result1

    def test_contains(self):
        ''' Test for containment check '''
        # cases:
        # - interval contains itself
        # - interval contains another interval (True)
        # - interval contains another interval (False, intersecting)
        # - interval contains another interval (False, non-intersecting)
        # - interval contains np.ndarray (inside)
        # - interval contains np.ndarray (on boundary)
        # - interval contains np.ndarray (outside)
        # - interval contains zonotope only center (True)
        # - interval contains zonotope only center (False)
        # - interval contains zonotope (True)
        # - interval contains zonotope (False)

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        lower_inside = np.array([-1., 0.])
        upper_inside = np.array([2., 3.5])
        lower_outside = np.array([-10., -8.])
        upper_outside = np.array([-4., 2.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower_inside, ub = upper_inside)
        I3 = Interval(lb = lower_outside, ub = upper)
        I4 = Interval(lb = lower_outside, ub = upper_outside)
        # init zonotopes
        center1 = np.array([1., 2.])
        center2 = np.array([0., 5.])
        generators = np.array([[1., 0.],[-1., 1.]])
        Z1 = Zonotope(c = center1)
        Z2 = Zonotope(c = center1, G = generators)
        Z3 = Zonotope(c = center2)
        Z4 = Zonotope(c = center2, G = generators)

        # check containment
        assert I1.contains(I1)
        assert I1.contains(I2)
        assert not I1.contains(I3)
        assert not I1.contains(I4)
        assert I1.contains(lower + np.array([1., 0.]))
        assert I1.contains(lower)
        assert not I1.contains(lower + np.array([-1., 0.]))
        assert I1.contains(Z1)
        assert I1.contains(Z2)
        assert not I1.contains(Z3)
        assert not I1.contains(Z4)

    def test_convex_hull(self):
        ''' Test for convex hull '''
        # cases:
        # - interval x itself
        # - interval x np.ndarray (inside)
        # - interval x np.ndarray (boundary)
        # - interval x np.ndarray (outside)
        # - interval x interval (non-intersecting)
        # - interval x zonotope (mode = outer)

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        I1 = Interval(lb = lower, ub = upper)
        v = np.array([-2., -1.])
        lower2 = np.array([8., 10.])
        upper2 = np.array([11., 15.])
        I2 = Interval(lb = lower2, ub = upper2)
        # init zonotope
        center = np.array([6., 8.])
        generators = np.array([[1., -1.],[-1., 0.]])
        Z = Zonotope(c = center, G = generators)

        # compute convex hull
        result1 = I1.convex_hull(I1)
        result2 = I1.convex_hull(I1.center())
        result3 = I1.convex_hull(lower)
        result4 = I1.convex_hull(lower + v)
        result5 = I1.convex_hull(I2)
        result6 = I1.convex_hull(Z)

        # manual computation
        true_result1 = I1
        true_result2 = I1
        true_result3 = I1
        true_result4 = Interval(lb = lower + v, ub = upper)
        true_result5 = Interval(lb = lower, ub = upper2)
        true_result6 = Interval(lb = lower, ub = np.array([8., 9.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6

        with self.assertRaises(NotImplementedError):
            # exact convex hull generally not an Interval
            I1 = Interval(lb = np.array([-1., 0.]), ub = np.array([1., 0.]))
            I2 = Interval(lb = np.array([3., -4.]), ub = np.array([4., -4.]))
            I1.convex_hull(I2, mode='exact')

    def test_cos(self):
        ''' Test for cosine '''
        # cases:
        # - diameter larger than 2*pi
        # - diameter is zero
        # - interval in monotonically decreasing region
        # - interval over valley at -1
        # - interval in monotonically increasing region
        # - interval over peak at +1

        # init interval
        # x:        0, pi/6,      pi/4,      pi/3, pi/2
        # cos(x):   0, sqrt(3)/2, 1/sqrt(2), 1/2,  0
        lower = np.array([-10., 0., np.pi/6., np.pi/2., 3.*np.pi/2., 3.*np.pi/2.])
        upper = np.array([10., 0., np.pi/4., 5.*np.pi/4., 7.*np.pi/4., 11.*np.pi/4.])
        I = Interval(lb = lower, ub = upper)
        # shift interval by various multiples of 2*pi
        shift = np.array([4., -5., 1., 0., 2., -1.]) * 2*np.pi
        I_shifted = I + shift

        # compute sine
        result1 = I.cos()
        result2 = I_shifted.cos()

        # manual computation
        true_result1 = Interval(lb = np.array([-1., 1., 1/np.sqrt(2), -1., 0., -1/np.sqrt(2)]),\
                                ub = np.array([1., 1., np.sqrt(3)/2., 0., 1/np.sqrt(2.), 1.]))
        true_result2 = true_result1

        # check result
        assert result1 == true_result1
        assert result2 == true_result2

    def test_diameter(self):
        ''' Test for computation of diameter '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # compute diameter
        result1 = I1.diameter()
        result2 = I2.diameter()
        result3 = I3.diameter()
        
        # manual computation
        true_result1 = np.array([5., 5.])
        true_result2 = np.array([0., 5.])
        true_result3 = np.zeros(2)

        # check results
        assert np.array_equal(result1, true_result1)
        assert np.array_equal(result2, true_result2)
        assert np.array_equal(result3, true_result3)

    def test_dot(self):
        ''' Test for dot product '''
        # cases:
        # - interval x vector
        # - interval x interval

        # init intervals
        lower1 = np.array([-2., -1., 1.])
        upper1 = np.array([1., -1., 2.])
        lower2 = np.array([2., -1., -4.])
        upper2 = np.array([3., 1., -3.])
        I1 = Interval(lb = lower1, ub = upper1)
        I2 = Interval(lb = lower2, ub = upper2)
        v = np.array([3., -1., 0.])

        # compute dot product
        result1 = I1.dot(v)
        result2 = I1.dot(I2)
        result3 = I2.dot(I1)

        # manual computation
        true_result1 = Interval(lb = np.array([-5.]), ub = np.array([4.]))
        true_result2 = Interval(lb = np.array([-15.]), ub = np.array([1.]))
        true_result3 = true_result2

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_intersects(self):
        ''' Test for intersection check '''
        # cases:
        # - interval x vector
        # - interval x interval (itself)
        # - interval x interval (below)
        # - interval x interval (intersects from below)
        # - interval x interval (contained)
        # - interval x interval (intersects from above)
        # - interval x interval (above)
        # - interval x zonotope

        # init intervals
        lower = np.array([-3., 0., 1.])
        upper = np.array([-1., 2., 5.])
        I = Interval(lb = lower, ub = upper)
        vector = np.array([-1., 1., 2.])
        vector_outside = np.array([-4., 1., 2.])
        lower_below = np.array([-4., 3., -3.])
        upper_below = np.array([-3.5, 4., -2.])
        I_below = Interval(lb = lower_below, ub = upper_below)
        upper_intersects_below = np.array([-2.5, 4., -2.])
        I_intersects_below = Interval(lb = lower_below, ub = upper_intersects_below)
        lower_contained = np.array([-2.5, 3., -3.])
        upper_contained = np.array([-1.5, 4., -2.])
        I_contained = Interval(lb = lower_contained, ub = upper_contained)
        lower_intersects_above = np.array([-1.5, 4., -2.])
        upper_intersects_above = np.array([2., 4., -2.])
        I_intersects_above = Interval(lb = lower_intersects_above, ub = upper_intersects_above)
        lower_above = np.array([-0.5, 4., -2.])
        upper_above = np.array([2., 4., -2.])
        I_above = Interval(lb = lower_above, ub = upper_above)
        Z = Zonotope(c = np.array([-2., 0., 1.]))

        # check intersection
        assert I.intersects(vector)
        assert not I.intersects(vector_outside)
        assert I.intersects(I)
        assert not I.intersects(I_below)
        assert I.intersects(I_intersects_below)
        assert I.intersects(I_contained)
        assert I.intersects(I_intersects_above)
        assert not I.intersects(I_above)
        assert I.intersects(Z)

    def test_interval(self):
        ''' Test for conversion to interval '''
        # cases:
        # - interval

        # init interval
        lower = np.array([1., 2.])
        upper = np.array([3., 4.])
        I = Interval(lb = lower, ub = upper)

        # convert to interval
        result1 = Interval(**I.interval())

        # manual computation
        true_result1 = I

        # check result
        assert result1 == true_result1

    def test_log(self):
        ''' Test for natural logarithm '''
        # cases:
        # - single point
        # - full-dimensional

        # init interval
        lower = np.array([1, np.exp(2), np.exp(3)])
        upper = np.array([np.exp(1), np.exp(4), np.exp(5)])
        I1 = Interval(lb = lower)
        I2 = Interval(lb = lower, ub = upper)

        # compute natural logarithm
        result1 = I1.log()
        result2 = I2.log()

        # manual computation
        true_result1 = Interval(lb = np.array([0., 2., 3.]))
        true_result2 = Interval(lb = np.array([0., 2., 3.]), ub = np.array([1., 4., 5.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # lower bound <= 0
            I = Interval(lb = np.array([-1.]), ub = np.array([1.]))
            I.log()

    def test_log10(self):
        ''' Test for logarithm with base 10 '''
        # cases:
        # - single point
        # - full-dimensional

        # init interval
        lower = np.array([1, 100, 1000])
        upper = np.array([10, 10000, 100000])
        I1 = Interval(lb = lower)
        I2 = Interval(lb = lower, ub = upper)

        # compute natural logarithm
        result1 = I1.log10()
        result2 = I2.log10()

        # manual computation
        true_result1 = Interval(lb = np.array([0., 2., 3.]))
        true_result2 = Interval(lb = np.array([0., 2., 3.]), ub = np.array([1., 4., 5.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # lower bound <= 0
            I = Interval(lb = np.array([-1.]), ub = np.array([1.]))
            I.log10()

    def test_minkowski_sum(self):
        ''' Test for Minkowski sum '''
        # cases:
        # - interval x vector
        # - interval x interval
        # - interval x zonotope

        # init interval
        lower1 = np.array([-2., 3., 0.])
        upper1 = np.array([4., 9., 2.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([1., -2., 4.])
        lower2 = np.array([-2., 1., 0.])
        upper2 = np.array([1., 5., 1.])
        I2 = Interval(lb = lower2, ub = upper2)
        center = np.array([1., 0., 2.])
        generators = np.array([[1., -2., 0.],[0., 1., 1.],[0., 0., 1.]])
        Z = Zonotope(c = center, G = generators)

        # Minkowski sum
        result1 = I1.minkowski_sum(vector)
        result2 = I1.minkowski_sum(I2)
        result3 = I1.minkowski_sum(Z)

        # manual computation
        true_result1 = Interval(lb = np.array([-1., 1., 4.]),\
                                ub = np.array([5., 7., 6.]))
        true_result2 = Interval(lb = np.array([-4., 4., 0.]),\
                                ub = np.array([5., 14., 3.]))
        true_result3 = Interval(lb = np.array([-4., 1., 1.]),\
                                ub = np.array([8., 11., 5.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_minkowski_difference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - interval x vector
        # - interval x interval
        # - interval x zonotope

        # init interval
        lower1 = np.array([-2., 3., 0.])
        upper1 = np.array([4., 9., 2.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([1., -2., 4.])
        lower2 = np.array([-2., 1., 0.])
        upper2 = np.array([1., 5., 1.])
        I2 = Interval(lb = lower2, ub = upper2)
        center = np.array([1., 0., 2.])
        generators = np.array([[1., -2., 0.],[0., 1., 1.],[0., 0., 1.]])
        Z = Zonotope(c = center, G = generators)

        # Minkowski difference
        result1 = I1.minkowski_difference(vector)
        result2 = I1.minkowski_difference(I2)
        result3 = I1.minkowski_difference(Z)

        # manual computation
        true_result1 = Interval(lb = np.array([-3., 5., -4.]),\
                                ub = np.array([3., 11., -2.]))
        true_result2 = Interval(lb = np.array([0., 2., 0.]),\
                                ub = np.array([3., 4., 1.]))
        true_result3 = Interval(lb = np.array([0., 5., -1.]),\
                                ub = np.array([0., 7., -1.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

        with self.assertRaises(exceptions.EmptySetError):
            # subtrahend too large -> empty set
            I1 = Interval(lb = np.array([-1., 0.]), ub = np.array([1., 0.]))
            I2 = Interval(lb = np.array([3., -4.]), ub = np.array([4., -2.]))
            I1.minkowski_difference(I2)

    def test_matmul(self):
        ''' Test for linear map '''
        # cases:
        # - interval x 0
        # - interval x identity
        # - interval x 

        # init interval
        lower = np.array([-4., -1., -1., 0., 4.])
        upper = np.array([-2., 0., 2., 3., 6.])
        I = Interval(lb = lower, ub = upper)

        # init matrices
        M0 = np.zeros((3,5))
        M1 = np.eye(I.dimension)
        M2 = np.array([[-2., 0., 0., 1., 2.],[1., -1., 2., 0., 1.],[-3., 2., 8., 0., -2.],[1., 1., -1., 0., 0.]])

        # compute linear map
        result1 = I.matmul(M0)
        result2 = I.matmul(M1)
        result3 = I.matmul(M2)

        # manual computation
        true_result1 = Interval(lb = np.zeros(3), ub = np.zeros(3))
        true_result2 = I
        true_result3 = Interval(lb = np.array([12., -2., -16., -7.]), ub = np.array([23., 9., 20., -1.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
    
    def test_plot(self):
        ''' Test for plotting '''
        # cases:
        # - single point
        # - degenerate (line)
        # - full-dimensional

        # init intervals
        lower = np.array([-2., 1., -1., 0.])
        upper = np.array([4., 2., -1., 0.])
        I = Interval(lb = lower, ub = upper)

        # call plot (interactive mode to avoid blocking)
        with plt.ion():
            I.plot(axis = (0,1))
            plt.close()
            I.plot(axis = (1,2))
            plt.close()
            I.plot(axis = (2,3))
            plt.close()

        # check exceptions
        with self.assertRaises(ValueError):
            # plot on too many dimensions
            I.plot(axis = (0, 1, 2, 3))

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - projection on all axes
        # - projection on multiple axes
        # - projection on single axis

        # init interval
        lower = np.array([-2., -1., 0., -4.])
        upper = np.array([3., 4., 5., 2.])
        I = Interval(lb = lower, ub = upper)
        
        # subspaces
        subspace_all = (0, 1, 2, 3)
        subspace_mult = (1, 3)
        subspace_single = (2,)

        # projections
        result1 = I.project(axis = subspace_all)
        result2 = I.project(axis = subspace_mult)
        result3 = I.project(axis = subspace_single)

        # manual computation
        true_result1 = I
        true_result2 = Interval(lb = np.array([-1., -4.]), ub = np.array([4., 2.]))
        true_result3 = Interval(lb = np.array([0.]), ub = np.array([5.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_reduce(self):
        ''' Test for representation size reduction '''
        # cases:
        # - interval

        # init interval
        I = Interval(lb = np.array([1.]), ub = np.array([2.]))

        # reduction
        result1 = I.reduce()

        # manual computation
        true_result1 = I

        # check results
        assert result1 == true_result1

    def test_represents(self):
        ''' Test for representation check '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # check results
        assert I1.represents('Interval')
        assert I1.represents('Zonotope')
        assert I2.represents('Zonotope')
        assert I3.represents('Zonotope')

    def test_sin(self):
        ''' Test for sine '''
        # cases:
        # - diameter larger than 2*pi
        # - diameter is zero
        # - interval in monotonically increasing region
        # - interval over peak at +1
        # - interval in monotonically decreasing region
        # - interval over valley at -1

        # init interval
        # x:        0, pi/6, pi/4,      pi/3,      pi/2
        # sin(x):   0, 1/2,  1/sqrt(2), sqrt(3)/2, 1
        lower = np.array([-10., 0., np.pi/6., np.pi/4., 3.*np.pi/4., 7.*np.pi/6.])
        upper = np.array([10., 0., np.pi/4., 3.*np.pi/4., 5.*np.pi/4., 7.*np.pi/4.])
        I = Interval(lb = lower, ub = upper)
        # shift interval by various multiples of 2*pi
        shift = np.array([4., -5., 1., 0., 2., -1.]) * 2*np.pi
        I_shifted = I + shift

        # compute sine
        result1 = I.sin()
        result2 = I_shifted.sin()

        # manual computation
        true_result1 = Interval(lb = np.array([-1., 0., 0.5, 1/np.sqrt(2.), -1/np.sqrt(2.), -1.]),\
                                ub = np.array([1., 0., 1/np.sqrt(2.), 1, 1/np.sqrt(2.), -0.5]))
        true_result2 = true_result1
        
        # check result
        assert result1 == true_result1
        assert result2 == true_result2

    def test_sqrt(self):
        ''' Test for square root '''
        # cases:
        # - single point
        # - full-dimensional

        # init interval
        lower = np.array([4., 16., 0.])
        upper = np.array([9., 36., 16.])
        I1 = Interval(lb = lower)
        I2 = Interval(lb = lower, ub = upper)

        # compute square root
        result1 = I1.sqrt()
        result2 = I2.sqrt()

        # manual computation
        true_result1 = Interval(lb = np.array([2., 4., 0.]))
        true_result2 = Interval(lb = np.array([2., 4., 0.]), ub = np.array([3., 6., 4.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # lower bound < 0
            I = Interval(lb = np.array([-1.]), ub = np.array([1.]))
            I.sqrt()

    def test_support_function(self):
        ''' Test for support function evaluation '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # evaluate support function
        value1, vector1 = I1.support_function(np.array([1., 0.]))
        value2, vector2 = I1.support_function(np.array([2., 0.]))
        value3, vector3 = I1.support_function(np.array([1., 1.]))
        value4, vector4 = I2.support_function(np.array([-1., 0.]))
        value5, vector5 = I3.support_function(np.array([1., 0.]))
        value6, vector6 = I3.support_function(np.array([0., -1.]))

        assert value1 == 3 and I1.contains(vector1)
        assert value2 == 6 and I1.contains(vector2)
        assert value3 == 7 and np.array_equal(vector3, upper)
        assert value4 == 2 and vector4[0] == -2
        assert value5 == -2 and np.array_equal(vector5, lower)
        assert value6 == 1 and np.array_equal(vector6, lower)

    def test_symm(self):
        ''' Test for symmetric interval '''
        # cases:
        # - interval (single point)
        # - all-positive interval
        # - mixed-bounds interval

        # init intervals
        vector = np.array([-2., 1., 0., 4.])
        I1 = Interval(lb = vector, ub = vector)
        lower_allpos = np.array([2., 4., 3.])
        upper_allpos = np.array([4., 6., 3.])
        I2 = Interval(lb = lower_allpos, ub = upper_allpos)
        lower_mixed = np.array([-3., 2., -1., 4.])
        upper_mixed = np.array([-1., 3., 1., 6.])
        I3 = Interval(lb = lower_mixed, ub = upper_mixed)

        # symmetric interval computation
        result1 = I1.symm()
        result2 = I2.symm()
        result3 = I3.symm()

        # manual computation
        true_result1 = Interval(lb = np.array([-2., -1., 0., -4.]),
                                ub = np.array([2., 1., 0., 4.]))
        true_result2 = Interval(lb = np.array([-4., -6., -3.]),
                                ub = np.array([4., 6., 3.]))
        true_result3 = Interval(lb = np.array([-3., -3., -1., -6.]),
                                ub = np.array([3., 3., 1., 6.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_tan(self):
        ''' Test for tangent '''
        # cases:
        # - interval

        # init interval
        lower = np.array([-np.pi/12., 0.])
        upper = np.array([np.pi/4., np.pi/3.])
        I = Interval(lb = lower, ub = upper)

        # compute tangent
        result1 = I.tan()

        # manual computation
        true_result1 = Interval(lb = np.array([-2. + np.sqrt(3.), 0.]),
                                ub = np.array([1., np.sqrt(3.)]))

        # check results
        assert result1 == true_result1

        # check exceptions
        with self.assertRaises(exceptions.OutOfBoundsError):
            # diameter too large -> jump
            I = Interval(lb = np.array([0.]), ub = np.array([5.]))
            I.tan()
        with self.assertRaises(exceptions.OutOfBoundsError):
            # tangent of upper < tangent of lower -> jump
            I = Interval(lb = np.array([1.5]), ub = np.array([2.]))
            I.tan()

    def test_vertices(self):
        ''' Test for vertex enumeration '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # compute vertices
        result1 = I1.vertices()
        result2 = I2.vertices()
        result3 = I3.vertices()

        # manual computation
        true_result1 = np.array([[-2., -2., 3., 3.],[-1., 4., -1., 4.]])
        true_result2 = np.array([[-2., -2.],[-1., 4.]])
        true_result3 = lower.reshape(2, 1)

        # check result
        assert comparison.compare_matrices(result1, true_result1)
        assert comparison.compare_matrices(result2, true_result2)
        assert comparison.compare_matrices(result3, true_result3)

    def test_volume(self):
        ''' Test for volume computation '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # compute volume
        result1 = I1.volume()
        result2 = I2.volume()
        result3 = I3.volume()

        # manual result
        true_result1 = 25.
        true_result2 = 0.
        true_result3 = 0.

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_zonotope(self):
        ''' Test for conversion from interval to zonotope '''
        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)

        # convert to zonotope (dictionary for constructor)
        result1 = Zonotope(**I1.zonotope())
        result2 = Zonotope(**I2.zonotope())
        result3 = Zonotope(**I3.zonotope())

        # manual computation
        true_result1 = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 2.5]]))
        true_result2 = Zonotope(c = np.array([-2., 1.5]), G = np.array([[0.],[2.5]]))
        true_result3 = Zonotope(c = lower)

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3

    def test_array_ufunc(self):
        ''' Test for overloading of right-operations '''
        # cases:
        # - number ** interval

        # init interval
        I = Interval(lb = np.array([0.]), ub = np.array([1.]))

        # check exceptions
        with self.assertRaises(NotImplementedError):
            np.array([1., 0.,]) ** I

    def test_check_interval_arithmetic(self):
        ''' Test for check function '''
        # cases:
        # interval x string

        # init interval
        I = Interval(lb = np.array([-1.]), ub = np.array([2.]))

        # check exceptions
        with self.assertRaises(TypeError):
            I._checkIntervalArithmetic("something")

if __name__ == '__main__':
    unittest.main()