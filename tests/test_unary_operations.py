
import unittest
import numpy as np
import continuoussets as cs
from continuoussets import (
    convert, represents,
    equals
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

        HP1 = cs.convert(I1, 'HPolyhedron')
        HP2 = cs.convert(I2, 'HPolyhedron')
        HP3 = cs.convert(I3, 'HPolyhedron')

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                                   b = np.array([-1., 2., 1., 0.]))
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                                   b = np.array([3., 2., 1., 0.]))
        true_result3 = HPolyhedron(A = np.array([[1.], [-1.]]),
                                   b = np.array([4., -1.]))
        
        assert cs.equals(HP1, true_result1)
        assert cs.equals(HP2, true_result2)
        assert cs.equals(HP3, true_result3)


        # cases:
        # - interval

        # init interval
        lower = np.array([1., 2.])
        upper = np.array([3., 4.])
        I = Interval(lb = lower, ub = upper)

        result1 = cs.convert(I, 'Interval')
        true_result1 = I

        assert cs.equals(result1, true_result1)


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
        result1 = cs.convert(I1, 'VPolytope')
        result2 = cs.convert(I2, 'VPolytope')
        result3 = cs.convert(I3, 'VPolytope')
        result4 = cs.convert(I4, 'VPolytope')

        # manual computation
        true_result1 = VPolytope(V = np.array([[-2., -1.], [-2., 4.], [3., -1.], [3., 4.]]))
        true_result2 = VPolytope(V = np.array([[-2., -1.],[-2., 4.]]))
        true_result3 = VPolytope(V = lower)
        true_result4 = VPolytope(V = np.array([[1.], [4.]]))

        # check result
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)


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
        result1 = cs.convert(I1, 'Zonotope')
        result2 = cs.convert(I2, 'Zonotope')
        result3 = cs.convert(I3, 'Zonotope')
        result4 = cs.convert(I4, 'Zonotope')

        # manual computation
        true_result1 = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 2.5]]))
        true_result2 = Zonotope(c = np.array([-2., 1.5]), G = np.array([0., 2.5]))
        true_result3 = Zonotope(c = lower)
        true_result4 = Zonotope(c = np.array([2.5]), G = np.array([[1.5]]))

        # check results
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)


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

        HP1 = cs.convert(Z1, 'HPolyhedron')
        HP2 = cs.convert(Z2, 'HPolyhedron')
        HP3 = cs.convert(Z3, 'HPolyhedron')
        HP4 = cs.convert(Z4, 'HPolyhedron')
        HP5 = cs.convert(Z5, 'HPolyhedron')
        HP6 = cs.convert(Z6, 'HPolyhedron')

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

        assert cs.equals(HP1, true_result1)
        assert cs.equals(HP2, true_result2)
        assert cs.equals(HP3, true_result3)
        assert cs.equals(HP4, true_result4)
        assert cs.equals(HP5, true_result5, rtol = 1e-5)
        assert cs.equals(HP6, true_result6)


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
        result1 = cs.convert(Z1, 'Interval')
        result2 = cs.convert(Z1, 'Interval', mode = 'inner')
        result3 = cs.convert(Z1, 'Interval')
        result4 = cs.convert(Z2, 'Interval')
        result5 = cs.convert(Z3, 'Interval', mode = 'outer')
        result6 = cs.convert(Z4, 'Interval')

        # manual computation
        true_result1 = Interval(lb = c, ub = c)
        true_result2 = true_result1
        true_result3 = true_result1
        true_result4 = Interval(lb = np.array([-2., -1.]), ub = np.array([4., 1.]))
        true_result5 = Interval(lb = np.array([0., -2.]), ub = np.array([2., 2.]))
        true_result6 = Interval(lb = np.array([-2.]), ub = np.array([6.]))

        # check results
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)
        assert cs.equals(result6, true_result6)

        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(Z3, 'Interval')

        # check exceptions
        with self.assertRaises(NotImplementedError):
            # inner approximation not supported in general case
            cs.convert(Z3, 'Interval', mode = 'inner')
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact conversion not supported in general case
            cs.convert(Z3, 'Interval', mode = 'exact')


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
        result1 = cs.convert(Z1, 'VPolytope')
        result2 = cs.convert(Z2, 'VPolytope')
        result3 = cs.convert(Z3, 'VPolytope')

        # true results 
        true_result1 = VPolytope(V = center)
        true_result2 = VPolytope(V = np.array([[1., -5.], [5., -3.], [3., 3.], [1., 5.], [-3., 3.], [-1., -3.]]))
        true_result3 = VPolytope(V = np.array([[-2.], [6.]]))

        # check result
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)


        # cases:
        # - center and generators

        # init zonotope
        center = np.array([1., 2.])
        generators = np.array([[0., -1.], [3., 4.]])
        Z = Zonotope(c = center, G = generators)

        # convert to zonotope
        result1 = cs.convert(Z, 'Zonotope')

        # manual computation
        true_result1 = Z

        # check result
        assert cs.equals(result1, true_result1)


        # cases:
        # - hpolyhedron
        A = np.array([[1., 0.], [-1., 1.], [-1., -1.]])
        b = np.array([1., 2., 1.])
        HP = HPolyhedron(A = A, b = b)

        result1 = cs.convert(HP, 'HPolyhedron')
        assert cs.equals(HP, result1)


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
        
        VP1 = cs.convert(HP1, 'VPolytope')
        VP2 = cs.convert(HP2, 'VPolytope')
        VP3 = cs.convert(HP3, 'VPolytope')

        true_result1 = VPolytope(V = np.array([[-1., 0.], [1., -2.], [1., 2.]]))
        true_result2 = VPolytope(V = np.array([[2.], [4.]]))
        true_result3 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))

        assert cs.equals(VP1, true_result1)
        assert cs.equals(VP2, true_result2)
        assert cs.equals(VP3, true_result3)

        with self.assertRaises(UnboundedSetError):
            cs.convert(HP4, 'VPolytope')


        # cases:
        # - bounded
        # - 1D
        # - zonotope-like
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.], [1.]]), b = np.array([4., -2., 7.]))
        HP3 = HPolyhedron(A = np.array([[0., -1.], [-1., -1.], [0.25, -0.5], [0., 1./3.], [1./7., 1./7.], [-0.25, 0.5]]),
                          b = np.ones(6))
        
        result1 = cs.convert(HP1, 'Zonotope', mode = 'outer')
        result2 = cs.convert(HP2, 'Zonotope')

        true_result2 = Zonotope(c = np.array([3.]), G = np.array([[1.]]))

        assert cs.contais(result1, HP1)
        assert cs.equals(result2, true_result2)

        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP1, 'Zonotope')
        with self.assertRaises(NotImplementedError):
            cs.convert(HP3, 'Zonotope')
        with self.assertRaises(NotImplementedError):
            cs.convert(HP1, 'Zonotope', mode = 'inner')


        # cases:
        # - single vertex
        # - degenerate
        # - non-degenerate
        # - 1D
        V_singlevertex = np.array([1., 0.])
        VP1 = VPolytope(V = V_singlevertex)
        VP2 = VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        VP3 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-2., -2.]]))
        VP4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        HP1 = cs.convert(VP1, 'HPolyhedron')
        HP2 = cs.convert(VP2, 'HPolyhedron')
        HP3 = cs.convert(VP3, 'HPolyhedron')
        HP4 = cs.convert(VP4, 'HPolyhedron')

        true_result1 = cs.convert(V_singlevertex, 'HPolyhedron')
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]]),
                                   b = np.array([1., 1., 1., -1.]))
        true_result3 = HPolyhedron(A = np.array([[1., -1.5], [1., 1.], [-1.5, 1.]]),
                                   b = np.array([1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 1.]))

        assert cs.equals(HP1, true_result1)
        assert cs.equals(HP2, true_result2)
        assert cs.equals(HP3, true_result3)
        assert cs.equals(HP4, true_result4)


        # cases:
        # - single vertex
        # - vpolytope that is an interval
        # - vpolytope that is not an interval
        # - 1D
        VP1 = VPolytope(V = np.array([1., 1.]))
        VP2 = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., 2.], [-1., 2.]]))
        VP3 = VPolytope(V = np.array([[-1., 0.], [0., -1.], [2., 1.]]))
        VP4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        result1 = cs.convert(VP1, 'Interval')
        result2 = cs.convert(VP2, 'Interval')
        result3 = cs.convert(VP3, 'Interval', mode = 'outer')
        result4 = cs.convert(VP4, 'Interval')

        I1 = Interval(lb = [1., 1.], ub = [1., 1.])
        I2 = Interval(lb = [-1., 0.], ub = [0., 2.])
        I3 = Interval(lb = [-1., -1.], ub = [2., 1.])
        I4 = Interval(lb = np.array([-1.]), ub = np.array([3.]))

        assert cs.equals(result1, I1)
        assert cs.equals(result2, I2)
        assert cs.equals(result3, I3)
        assert cs.equals(result4, I4)

        # unsupported conversions
        with self.assertRaises(NotImplementedError):
            cs.convert(VP3, 'Interval', mode = 'inner')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP3, 'Interval', mode = 'exact')
                          
                          
        # cases:
        # - single vertex
        # - non-degenerate set
        V1 = np.array([[2., 3., -1.]])
        VP1 = VPolytope(V = V1)
        V2 = np.array([[2., 1.], [-1., 2.], [0., -4.]])
        VP2 = VPolytope(V = V2)

        result1 = VP1, 'VPolytope'
        result2 = VP2, 'VPolytope'

        assert compare_matrices(result1['V'], V1)
        assert compare_matrices(result2['V'], V2)


        # cases:
        # - single vertex
        # - multiple vertices (not a zonotope)
        # - multiple vertices (is a zonotope)
        # - 1D
        V1 = np.array([[2., 3., -1.]])
        VP1 = VPolytope(V = V1)
        VP2 = VPolytope(V = np.array([[2., 1.], [-1., 2.], [0., -4.]]))
        VP3 = VPolytope(V = np.array([[1., -6.], [3., -2.], [3., 0.], [1., 2.], [-1., -2.], [-1., -4.]]))
        VP4 = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))

        result_1 = cs.convert(VP1, 'Zonotope')
        result_2 = cs.convert(VP2, 'Zonotope', mode = 'outer')
        result_4 = cs.convert(VP4, 'Zonotope')

        assert cs.equals(result1, VP1)
        assert cs.contains(result2, VP2)
        assert cs.equals(result1, VP4)

        # unsupported conversions
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP2, 'Zonotope', mode = 'exact')
        with self.assertRaises(NotImplementedError):
            cs.convert(VP2, 'Zonotope', mode = 'inner')
        with self.assertRaises(NotImplementedError):
            cs.convert(VP3, 'Zonotope')


    def test_Represents(self):
        # cases:
        # - interval: single point
        # - interval: degenerate
        # - interval: full-dimensional interval
        # - zonotope: single point
        # - zonotope: 1D
        # - zonotope: interval
        # - zonotope: non-degenerate
        # - vpolytope: single vertex
        # - vpolytope: interval
        # - vpolytope: zonotope
        # - vpolytope: uneven number of vertices (cannot be a zonotope or an interval)
        # - hpolyhedron: bounded
        # - hpolyhedron: 1D
        # - hpolyhedron: interval
        # - hpolyhedron: interval-like, but unbounded
        # - hpolyhedron: zonotope
        # - hpolyhedron: zonotope-like, but unbounded

        # init intervals
        I_singlepoint = Interval(lb = np.array([-2., -1.]))
        I_deg = Interval(lb = np.array([-2., -1.]), ub = np.array([-2., 4.]))
        I_nondeg = Interval(lb = np.array([-2., -1.]), ub = np.array([3., 4.]))
        
        # init zonotopes
        Z_singlepoint = Zonotope(c = np.array([1., 0.]))
        Z_1D = Zonotope(c = np.array([1.]), G = np.array([[2.], [-1.]]))
        Z_interval = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0.], [0., -1.], [2., 0.], [0., 0.]]))
        Z_nondeg = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0.], [-1., 1.]]))

        # init vpolytopes
        VP_singlepoint = VPolytope(V = np.array([2., 1.]))
        VP_interval = VPolytope(V = np.array([[-1., 0.], [2., 0.], [2., 1.], [-1., 1.]]))
        VP_zonotope = VPolytope(V = np.array([[4., -1., 1.], [2., -3., 1.], [6., 3., -1.], [4., 1., -1.],
                                       [-2., -3., 3.], [2., 3., 1.], [0., 1., 1.], [2., -3., 3.],
                                       [0., -5., 3.], [4., 1., 1.], [-2., -3., 5.], [-4., -5., 5.],
                                       [0., 1., 3.], [-2., -1., 3.]]))
        VP_nondeg = VPolytope(V = np.array([[2., 1.], [0., 2.], [-1., -2.]]))

        # init hpolyhedra
        HP_singlepoint = cs.convert(np.array([2., 1.]), 'HPolyhedron')
        HP_1D = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([4., -3.]))
        HP_interval = HPolyhedron(A = np.array([[1., 0.], [0., 2.], [-1., 0.], [0., -4], [1., 1.]]),
                                  b = np.array([3., 2., 5., 1., 30.]))
        HP_zonotope = HPolyhedron(A = np.array([[-0.5, -0.5], [0., -1/3.], [1./np.sqrt(2.), 1./np.sqrt(2.)], [0., 1.]]),
                                  b = np.array([1., 1., 0., -1.]))
        HP_nondeg = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                b = np.array([1., 1., 1.]))
        HP_interval_unb = HPolyhedron(A = np.array([[1., 0.], [0., 2.], [0., -4], [1., 1.]]),
                                      b = np.array([3., 2., 1., 30.]))
        HP_zonotope_unb = HPolyhedron(A = np.array([[1., 0.], [-1., 0.]]), b = np.ones(2))

        # check intervals
        assert cs.represents(I_singlepoint, 'ndarray')
        assert cs.represents(I_singlepoint, 'Interval')
        assert cs.represents(I_singlepoint, 'Zonotope')
        assert cs.represents(I_singlepoint, 'VPolytope')
        assert cs.represents(I_singlepoint, 'HPolyhedron')

        assert not cs.represents(I_deg, 'ndarray')
        assert cs.represents(I_deg, 'Interval')
        assert cs.represents(I_deg, 'Zonotope')
        assert cs.represents(I_deg, 'VPolytope')
        assert cs.represents(I_deg, 'HPolyhedron')

        assert not cs.represents(I_nondeg, 'ndarray')
        assert cs.represents(I_nondeg, 'Interval')
        assert cs.represents(I_nondeg, 'Zonotope')
        assert cs.represents(I_nondeg, 'VPolytope')
        assert cs.represents(I_nondeg, 'HPolyhedron')

        # check zonotopes
        assert cs.represents(Z_singlepoint, 'ndarray')
        assert cs.represents(Z_singlepoint, 'Interval')
        assert cs.represents(Z_singlepoint, 'Zonotope')
        assert cs.represents(Z_singlepoint, 'VPolytope')
        assert cs.represents(Z_singlepoint, 'HPolyhedron')

        assert not cs.represents(Z_1D, 'ndarray')
        assert cs.represents(Z_1D, 'Interval')
        assert cs.represents(Z_1D, 'Zonotope')
        assert cs.represents(Z_1D, 'VPolytope')
        assert cs.represents(Z_1D, 'HPolyhedron')

        assert not cs.represents(Z_interval, 'ndarray')
        assert cs.represents(Z_interval, 'Interval')
        assert cs.represents(Z_interval, 'Zonotope')
        assert cs.represents(Z_interval, 'VPolytope')
        assert cs.represents(Z_interval, 'HPolyhedron')

        assert not cs.represents(Z_nondeg, 'ndarray')
        assert not cs.represents(Z_nondeg, 'Interval')
        assert cs.represents(Z_nondeg, 'Zonotope')
        assert cs.represents(Z_nondeg, 'VPolytope')
        assert cs.represents(Z_nondeg, 'HPolyhedron')

        # check vpolytopes
        assert cs.represents(VP_singlepoint, 'ndarray')
        assert cs.represents(VP_singlepoint, 'Interval')
        assert cs.represents(VP_singlepoint, 'Zonotope')
        assert cs.represents(VP_singlepoint, 'VPolytope')
        assert cs.represents(VP_singlepoint, 'HPolyhedron')

        assert not cs.represents(VP_interval, 'ndarray')
        assert cs.represents(VP_interval, 'Interval')
        assert cs.represents(VP_interval, 'Zonotope')
        assert cs.represents(VP_interval, 'VPolytope')
        assert cs.represents(VP_interval, 'HPolyhedron')

        assert not cs.represents(VP_zonotope, 'ndarray')
        assert not cs.represents(VP_zonotope, 'Interval')
        assert cs.represents(VP_zonotope, 'Zonotope')
        assert cs.represents(VP_zonotope, 'VPolytope')
        assert cs.represents(VP_zonotope, 'HPolyhedron')

        assert not cs.represents(VP_nondeg, 'ndarray')
        assert not cs.represents(VP_nondeg, 'Interval')
        assert not cs.represents(VP_nondeg, 'Zonotope')
        assert cs.represents(VP_nondeg, 'VPolytope')
        assert cs.represents(VP_nondeg, 'HPolyhedron')
        
        # check hpolyhedra
        assert cs.represents(HP_singlepoint, 'ndarray')
        assert cs.represents(HP_singlepoint, 'Interval')
        assert cs.represents(HP_singlepoint, 'Zonotope')
        assert cs.represents(HP_singlepoint, 'VPolytope')
        assert cs.represents(HP_singlepoint, 'HPolyhedron')

        assert not cs.represents(HP_1D, 'ndarray')
        assert cs.represents(HP_1D, 'Interval')
        assert cs.represents(HP_1D, 'Zonotope')
        assert cs.represents(HP_1D, 'VPolytope')
        assert cs.represents(HP_1D, 'HPolyhedron')
        
        assert not cs.represents(HP_zonotope, 'ndarray')
        assert cs.represents(HP_interval, 'Interval')
        assert cs.represents(HP_interval, 'Zonotope')
        assert cs.represents(HP_interval, 'VPolytope')
        assert cs.represents(HP_interval, 'HPolyhedron')

        assert not cs.represents(HP_zonotope, 'ndarray')
        assert not cs.represents(HP_zonotope, 'Interval')
        assert cs.represents(HP_zonotope, 'Zonotope')
        assert cs.represents(HP_zonotope, 'VPolytope')
        assert cs.represents(HP_zonotope, 'HPolyhedron')

        assert not cs.represents(HP_nondeg, 'ndarray')
        assert not cs.represents(HP_nondeg, 'Interval')
        assert not cs.represents(HP_nondeg, 'Zonotope')
        assert cs.represents(HP_nondeg, 'HPolyhedron')
        assert cs.represents(HP_nondeg, 'VPolytope')
        
        assert not cs.represents(HP_interval_unb, 'Interval')
        assert not cs.represents(HP_interval_unb, 'Zonotope')
        assert not cs.represents(HP_interval_unb, 'VPolytope')
        assert not cs.represents(HP_zonotope_unb, 'Interval')
        assert not cs.represents(HP_zonotope_unb, 'Zonotope')
        assert not cs.represents(HP_zonotope_unb, 'VPolytope')


if __name__ == '__main__':
    unittest.main()
