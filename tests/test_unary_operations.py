
import unittest
import numpy as np
import continuoussets as cs
from continuoussets import (
    convert, represents
)
from continuoussets import (
    Interval, Zonotope, VPolytope, HPolyhedron
)
from continuoussets.utils.exceptions import ExactEvaluationImpossibleError, UnboundedSetError
from continuoussets.utils.comparison import compare_matrices


class TestUnaryOperations(unittest.TestCase):

    def test_Convert(self):
        # cases:
        # - degenerate
        # - non-degenerate
        # - 1D
        I1 = Interval(lb = np.array([-1., 0.]), ub = np.array([-1., 2.]))
        I2 = Interval(lb = np.array([-1., 0.]), ub = np.array([3., 2.]))
        I3 = Interval(lb = np.array([1.]), ub = np.array([4.]))

        HP1 = HPolyhedron(**I1.hpolyhedron())
        HP2 = HPolyhedron(**I2.hpolyhedron())
        HP3 = HPolyhedron(**I3.hpolyhedron())

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                                   b = np.array([-1., 2., 1., 0.]))
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                                   b = np.array([3., 2., 1., 0.]))
        true_result3 = HPolyhedron(A = np.array([[1.], [-1.]]),
                                   b = np.array([4., -1.]))
        
        assert HP1 == true_result1
        assert HP2 == true_result2
        assert HP3 == true_result3


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


        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point
        # - 1D

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)
        I4 = Interval(lb = np.array([1.]), ub = np.array([4.]))

        # compute vertices
        result1 = VPolytope(**I1.vpolytope())
        result2 = VPolytope(**I2.vpolytope())
        result3 = VPolytope(**I3.vpolytope())
        result4 = VPolytope(**I4.vpolytope())

        # manual computation
        true_result1 = VPolytope(V = np.array([[-2., -1.], [-2., 4.], [3., -1.], [3., 4.]]))
        true_result2 = VPolytope(V = np.array([[-2., -1.],[-2., 4.]]))
        true_result3 = VPolytope(V = lower)
        true_result4 = VPolytope(V = np.array([[1.], [4.]]))

        # check result
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4


        # cases:
        # - full-dimensional interval
        # - degenerate interval
        # - single point
        # - 1D

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        upper_degenerate = np.array([-2., 4.])
        I1 = Interval(lb = lower, ub = upper)
        I2 = Interval(lb = lower, ub = upper_degenerate)
        I3 = Interval(lb = lower)
        I4 = Interval(lb = np.array([1.]), ub = np.array([4.]))

        # convert to zonotope (dictionary for constructor)
        result1 = Zonotope(**I1.zonotope())
        result2 = Zonotope(**I2.zonotope())
        result3 = Zonotope(**I3.zonotope())
        result4 = Zonotope(**I4.zonotope())

        # manual computation
        true_result1 = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 2.5]]))
        true_result2 = Zonotope(c = np.array([-2., 1.5]), G = np.array([0., 2.5]))
        true_result3 = Zonotope(c = lower)
        true_result4 = Zonotope(c = np.array([2.5]), G = np.array([[1.5]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4


        # cases:
        # - only center
        # - center and generators (2D)
        # - center and generators (3D)
        # - degenerate (1D in 2D)
        # - degenerate (2D in 3D)
        # - 1D
        Z1 = Zonotope(c = np.array([1., 0.]))
        Z2 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0.], [-1., 1.], [2., 1.]]))
        Z3 = Zonotope(c = np.array([1., -1., 2.]),
                      G = np.array([[1., 1., 0.], [1., 2., -1.], [-2., 0., 1.], [-1., -1., 1.]]))
        Z4 = Zonotope(c = np.array([1., 0.]), G = np.array([[1., -2.]]))
        Z5 = Zonotope(c = np.array([2., -1., 1.]),
                      G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.,], [4., 2., -2.], [-1., 4., 5.]]))
        Z6 = Zonotope(c = np.array([2.]), G = np.array([[1.], [-2.], [0.], [-1.]]))

        HP1 = HPolyhedron(**Z1.hpolyhedron())
        HP2 = HPolyhedron(**Z2.hpolyhedron())
        HP3 = HPolyhedron(**Z3.hpolyhedron())
        HP4 = HPolyhedron(**Z4.hpolyhedron())
        HP5 = HPolyhedron(**Z5.hpolyhedron())
        HP6 = HPolyhedron(**Z6.hpolyhedron())

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]),
                                   b = np.array([1., 0., -1.]))
        true_result2 = HPolyhedron(A = np.array([[0., -0.5], [0.2, 0.2], [0.2, -0.4], [0., 0.5], [-1./3., -1./3.], [-1./3., 2./3.]]),
                                   b = np.array([1., 1., 1., 1., 1., 1.]))
        true_result3 = HPolyhedron(A = np.array([[-1., 0., -1.], [-1., 1., 0.], [-0.25, 0.25, 0.25], [-0.4, -0.2, -0.8],
                                                 [-1., -1., -2.], [-1., 1., -2.], [1./11., -1./11., 2./11.], [1./7., 1./7., 2./7.],
                                                 [2./13., 1./13., 4./13.], [0.25, -0.25, -0.25], [0.2, -0.2, 0.], [0.2, 0., 0.2]]),
                                   b = np.array([-1., 1., 1., -1., -1., -1., 1., 1., 1., 1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1./np.sqrt(5.), -2./np.sqrt(5.)],
                                                 [-1./np.sqrt(5.), 2./np.sqrt(5.)],
                                                 [2./np.sqrt(5.), 1./np.sqrt(5.)],
                                                 [-2./np.sqrt(5.), -1./np.sqrt(5.)]]),
                                   b = np.array([6./np.sqrt(5.), 4./np.sqrt(5.), 2./np.sqrt(5.), -2./np.sqrt(5.)]))
        true_result5 = HPolyhedron(A = np.array([[-1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
                                                 [-1./np.sqrt(2.), -1./np.sqrt(2.), 0.],
                                                 [-0.81650, -0.40825, 0.40825],
                                                 [0, 1./np.sqrt(2.), 1./np.sqrt(2.)],
                                                 [-0.80178, -0.53452, 0.26726],
                                                 [1./np.sqrt(2.), 0., -1./np.sqrt(2.)],
                                                 [1./np.sqrt(2.), 1./np.sqrt(2.), 0.],
                                                 [0.81650, 0.40825, -0.40825],
                                                 [0, -1./np.sqrt(2.), -1./np.sqrt(2.)],
                                                 [0.80178, 0.53452, -0.26726],
                                                 [-0.57735, 0.57735, -0.57735],
                                                 [0.57735, -0.57735, 0.57735]]),
                                   b = np.array([12.02082, 9.89949, 7.75672, 14.84924, 7.21605, 13.43503,
                                                 11.31371, 9.38971, 14.84924, 8.81962, -2.30940, 2.30940]))
        true_result6 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([6., 2.]))

        assert HP1 == true_result1
        assert HP2 == true_result2
        assert HP3 == true_result3
        assert HP4 == true_result4
        assert HP5.__eq__(true_result5, rtol = 1e-5)
        assert HP6 == true_result6


        # cases:
        # - only center
        # - center and generators (box)
        # - center and generators (not a box)
        # - 1D

        # init zonotopes
        c = np.array([1., 0.])
        G_axisaligned = np.array([[1., 0.], [0., -1.], [2., 0.], [0., 0.]])
        G_notaxisaligned = np.array([[1., -1.], [0., 1.]])
        Z1 = Zonotope(c = c)
        Z2 = Zonotope(c = c, G = G_axisaligned)
        Z3 = Zonotope(c = c, G = G_notaxisaligned)
        Z4 = Zonotope(c = np.array([2.]), G = np.array([[1.], [-2.], [0.], [1.]]))

        # convert to intervals (dictionary for constructor)
        result1 = Interval(**Z1.interval())
        result2 = Interval(**Z1.interval(mode = 'inner'))
        result3 = Interval(**Z1.interval())
        result4 = Interval(**Z2.interval())
        result5 = Interval(**Z3.interval(mode = 'outer'))
        result6 = Interval(**Z4.interval())

        # manual computation
        true_result1 = Interval(lb = c, ub = c)
        true_result2 = true_result1
        true_result3 = true_result1
        true_result4 = Interval(lb = np.array([-2., -1.]), ub = np.array([4., 1.]))
        true_result5 = Interval(lb = np.array([0., -2.]), ub = np.array([2., 2.]))
        true_result6 = Interval(lb = np.array([-2.]), ub = np.array([6.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6

        with self.assertRaises(ExactEvaluationImpossibleError):
            Interval(**Z3.interval())

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # inner approximation not supported in general case
            Z3.interval(mode='inner')
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact conversion not supported in general case
            Z3.interval(mode='exact')


        # cases:
        # - only center
        # - center and generators
        # - 1D

        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 1.], [-1., 3.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)
        Z3 = Zonotope(c = np.array([2.]), G = np.array([[1.], [-2.], [0.], [-1.]]))

        # convert to vpolytope
        result1 = VPolytope(**(Z1.vpolytope()))
        result2 = VPolytope(**(Z2.vpolytope()))
        result3 = VPolytope(**(Z3.vpolytope()))

        # true results 
        true_result1 = VPolytope(V = center)
        true_result2 = VPolytope(V = np.array([[1., -5.], [5., -3.], [3., 3.], [1., 5.], [-3., 3.], [-1., -3.]]))
        true_result3 = VPolytope(V = np.array([[-2.], [6.]]))

        # check result
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3


        # cases:
        # - center and generators

        # init zonotope
        center = np.array([1., 2.])
        generators = np.array([[0., -1.], [3., 4.]])
        Z = Zonotope(c = center, G = generators)

        # convert to zonotope
        result1 = Zonotope(**Z.zonotope())

        # manual computation
        true_result1 = Z

        # check result
        assert result1 == true_result1


        # cases:
        # - hpolyhedron
        A = np.array([[1., 0.], [-1., 1.], [-1., -1.]])
        b = np.array([1., 2., 1.])
        HP = HPolyhedron(A = A, b = b)

        d = HP.hpolyhedron()
        assert np.array_equal(d['A'], A)
        assert np.array_equal(d['b'], b)


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

        with self.assertRaises(UnboundedSetError):
            VPolytope(**HP4.vpolytope())


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

        with self.assertRaises(ExactEvaluationImpossibleError):
            HP1.zonotope()
        with self.assertRaises(NotImplementedError):
            HP3.zonotope()
        with self.assertRaises(NotImplementedError):
            HP1.zonotope(mode = 'inner')


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

        true_result1 = cs.convert(V_singlevertex, 'HPolyhedron')
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]]),
                                   b = np.array([1., 1., 1., -1.]))
        true_result3 = HPolyhedron(A = np.array([[1., -1.5], [1., 1.], [-1.5, 1.]]),
                                   b = np.array([1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 1.]))

        assert HP_1 == true_result1
        assert HP_2 == true_result2
        assert HP_3 == true_result3
        assert HP_4 == true_result4


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
        with self.assertRaises(ExactEvaluationImpossibleError):
            VP_3.interval(mode = 'exact')
                          
                          
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2., 3., -1.]])
        VP_1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP_2 = VPolytope(V = V2)

        result1 = VP_1.vpolytope()
        result2 = VP_2.vpolytope()

        assert compare_matrices(result1['V'], V1)
        assert compare_matrices(result2['V'], V2)


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
        with self.assertRaises(ExactEvaluationImpossibleError):
            VP_2.zonotope(mode = 'exact')
        with self.assertRaises(NotImplementedError):
            VP_2.zonotope(mode = 'inner')
        with self.assertRaises(NotImplementedError):
            VP_3.zonotope()


    def test_Represents(self):
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
        assert not I2.represents('Point')
        assert I3.represents('Zonotope')
        assert I3.represents('VPolytope')
        assert I3.represents('HPolyhedron')
        assert I3.represents('Point')


        # cases:
        # - only center
        # - center and generators (box)
        # - center and generators (not a box)
        # - 1D

        # init zonotopes
        c = np.array([1., 0.])
        G_axisaligned = np.array([[1., 0.], [0., -1.], [2., 0.], [0., 0.]])
        G_notaxisaligned = np.array([[1., 0.], [-1., 1.]])
        Z1 = Zonotope(c = c)
        Z2 = Zonotope(c = c, G = G_axisaligned)
        Z3 = Zonotope(c = c, G = G_notaxisaligned)
        Z4 = Zonotope(c = np.array([1.]), G = np.array([[2.], [-1.]]))

        # check representation
        assert Z1.represents('Interval')
        assert Z1.represents('Point')
        assert Z2.represents('Interval')
        assert not Z3.represents('Point')
        assert not Z3.represents('Interval')
        assert Z1.represents('Zonotope')
        assert Z3.represents('VPolytope')
        assert Z3.represents('HPolyhedron')
        assert Z4.represents('Interval')
        assert Z4.represents('HPolyhedron')


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
        HP6 = cs.convert(np.array([2., 1.]), 'HPolyhedron')
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
