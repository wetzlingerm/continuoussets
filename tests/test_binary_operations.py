
import unittest
import numpy as np
import continuoussets as cs

from continuoussets import (
    cartesian_product, convex_hull, intersection, minkowski_difference, minkowski_sum,
    contains, equals, intersects
)
from continuoussets import (
    Interval, Zonotope, VPolytope, HPolyhedron
)
from continuoussets.utils.exceptions import EmptySetError, UnboundedSetError, ExactEvaluationImpossibleError
from continuoussets.utils.comparison import compare_matrices


class TestBinaryOperations(unittest.TestCase):

    def test_CartesianProduct(self):
        # cases:
        # - interval x interval
        # - interval x np.ndarray
        # - interval x zonotope
        # - interval x vpolytope
        # - interval x hpolyhedron

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        I1 = Interval(lb = lower, ub = upper)
        # init zonotope
        Z1 = Zonotope(c = np.array([1., 0.]), G = np.array([[-1., 0.],[0., 2.]]))
        Z2 = Zonotope(c = np.array([1., 0.]), G = np.array([[-1., 0.],[1., 2.]]))
        # init vpolytopes
        VP1 = VPolytope(V = np.array([[1.], [2.]]))
        VP2 = VPolytope(V = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]]))
        VP3 = VPolytope(V = np.array([[0., 1.], [1., 0.]]))
        # init hpolyhedron
        HP1 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([4., -2.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))

        # compute Cartesian product
        result1 = I1.cartesian_product(I1)
        result2 = I1.cartesian_product(lower)
        result3 = I1.cartesian_product(Z1)
        result4 = I1.cartesian_product(Z2, mode = 'outer')
        result5 = I1.cartesian_product(VP1)
        result6 = I1.cartesian_product(VP2)
        result7 = I1.cartesian_product(HP1)
        result8 = I1.cartesian_product(HP2, mode = 'outer')

        # manual computation
        true_result1 = Interval(lb = np.hstack((lower, lower)),\
                                ub = np.hstack((upper, upper)))
        true_result2 = Interval(lb = np.hstack((lower, lower)),\
                                ub = np.hstack((upper, lower)))
        true_result3 = Interval(lb = np.hstack((lower, np.array([0., -2.]))),\
                                ub = np.hstack((upper, np.array([2., 2.]))))
        true_result4 = Interval(lb = np.hstack((lower, np.array([-1., -2.]))),\
                                ub = np.hstack((upper, np.array([3., 2.]))))
        true_result5 = Interval(lb = np.hstack((lower, np.array([1.]))),\
                                ub = np.hstack((upper, np.array([2.]))))
        true_result6 = Interval(lb = np.hstack((lower, np.array([0., 0.]))),\
                                ub = np.hstack((upper, np.array([1., 1.]))))
        true_result7 = Interval(lb = np.hstack((lower, np.array([2.]))),\
                                ub = np.hstack((upper, np.array([4.]))))
        true_result8 = Interval(lb = np.hstack((lower, np.array([-1., -2.]))),\
                                ub = np.hstack((upper, np.array([1., 2.]))))
        
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8 == true_result8

        with self.assertRaises(ExactEvaluationImpossibleError):
            # 'exact' not supported in general for Interval x Zonotope
            I1.cartesian_product(Z2)
        with self.assertRaises(ExactEvaluationImpossibleError):
            # 'exact' not supported in general for Interval x VPolytope
            I1.cartesian_product(VP3)
        with self.assertRaises(ExactEvaluationImpossibleError):
            # 'exact' not supported in general for Interval x HPolyhedron
            I1.cartesian_product(HP2)


        # cases:
        # - zonotope with only center x zonotope with only center
        # - zonotope with only center x vector
        # - zonotope with only center x zonotope
        # - zonotope x vector
        # - zonotope x zonotope with only center
        # - zonotope x zonotope
        # - zonotope x interval
        # - zonotope x vpolytope
        # - zonotope x hpolyhedron

        # init zonotopes
        center1 = np.array([-1., 0.])
        center2 = np.array([3.])
        generators1 = np.array([[2., 1.]])
        generators2 = np.array([[2.], [1.], [-1.], [0.]])
        Z1_onlycenter = Zonotope(c = center1)
        Z2_onlycenter = Zonotope(c = center2)
        Z3 = Zonotope(c = center1, G = generators1)
        Z4 = Zonotope(c = center2, G = generators2)
        # init interval, vpolytope, hpolyhedron
        I = Interval(lb = np.array([-1.]), ub = np.array([7.]))
        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -2.]]))
        VP2 = VPolytope(V = center2)
        HP = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        # compute Cartesian product
        result1 = Z1_onlycenter.cartesian_product(Z2_onlycenter)
        result2 = Z1_onlycenter.cartesian_product(center2)
        result3 = Z1_onlycenter.cartesian_product(Z4)
        result4 = Z3.cartesian_product(center2)
        result5 = Z3.cartesian_product(Z2_onlycenter)
        result6 = Z3.cartesian_product(Z4)
        result7 = Z1_onlycenter.cartesian_product(I)
        result8 = Z3.cartesian_product(VP1, mode = 'outer')
        result9 = Z3.cartesian_product(VP2)
        result10 = Z3.cartesian_product(HP, mode = 'outer')
        
        # manual computation
        centers_stacked = np.array([-1., 0., 3.])
        zero_generators2 = np.array([[0., 0., 2.], [0., 0., 1.], [0., 0., -1.]])
        generators1_zero = np.array([[2., 1., 0.]])
        generators1_generators2 = np.array([[2., 1., 0.], [0., 0., 2.], [0., 0., 1.], [0., 0., -1.]])
        true_result1 = Zonotope(c = centers_stacked)
        true_result2 = true_result1
        true_result3 = Zonotope(c = centers_stacked, G = zero_generators2)
        true_result4 = Zonotope(c = centers_stacked, G = generators1_zero)
        true_result5 = Zonotope(c = centers_stacked, G = generators1_zero)
        true_result6 = Zonotope(c = centers_stacked, G = generators1_generators2)
        true_result7 = Zonotope(c = centers_stacked, G = zero_generators2)
        true_result8 = HPolyhedron(A = np.array([[2./3., 1./3., 0., 0.],
                                                 [-2./7., -1./7., 0., 0.],
                                                 [-1., 2., 0., 0.],
                                                 [1., -2., 0., 0.],
                                                 [0., 0., 1., -1.],
                                                 [0., 0., 1., 1.],
                                                 [0., 0., -3., 1.]]),
                                   b = np.array([1., 1., 1., -1., 1., 1., 1.]))
        true_result9 = Zonotope(c = centers_stacked, G = generators1_zero)
        true_result10 = HPolyhedron(A = np.array([[2./3., 1./3., 0., 0.],
                                                 [-2./7., -1./7., 0., 0.],
                                                 [-1., 2., 0., 0.],
                                                 [1., -2., 0., 0.],
                                                 [0., 0., 1., 0.],
                                                 [0., 0., -1., -1.],
                                                 [0., 0., -1., 1.]]),
                                    b = np.array([1., 1., 1., -1., 1., 1., 1.]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8.contains(true_result8)
        assert result9 == true_result9
        assert result10.contains(true_result10)

        with self.assertRaises(ExactEvaluationImpossibleError):
            Z3.cartesian_product(VP1)
        with self.assertRaises(ExactEvaluationImpossibleError):
            Z3.cartesian_product(HP)

    
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

        assert compare_matrices(VP1_VP2.V, V1_V2)
        assert compare_matrices(VP1_I2.V, V1_V2)

    def test_Contains(self):
        # cases:
        # - interval x itself
        # - interval x another interval (True)
        # - interval x another interval (False, intersecting)
        # - interval x another interval (False, non-intersecting)
        # - interval x np.ndarray (inside)
        # - interval x np.ndarray (on boundary)
        # - interval x np.ndarray (outside)
        # - interval x zonotope only center (True)
        # - interval x zonotope only center (False)
        # - interval x zonotope (True)
        # - interval x zonotope (False)
        # - interval x vpolytope (True)
        # - interval x vpolytope (False)
        # - interval x hpolyhedron (True)
        # - interval x hpolyhedron (False)

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
        # init vpolytopes
        VP1 = VPolytope(V = np.array([[-1., 0.], [2., 1.], [0., 3.]]))
        VP2 = VPolytope(V = np.array([[-1., -2.], [1., 1.], [-2., 3.]]))
        # init hpolyhedra
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 2., 0.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.]]), b = np.array([1.]))

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
        assert I1.contains(VP1)
        assert not I1.contains(VP2)
        assert I1.contains(HP1)
        assert not I1.contains(HP2)


        # cases:
        # - only center x point (inside)
        # - only center x point (outside)
        # - full-dimensional x point (inside)
        # - full-dimensional x point (boundary)
        # - full-dimensional x point (outside)
        # - zonotope x zonotope (only center)
        # - zonotope x interval
        # - zonotope x vpolytope
        # - zonotope x hpolyhedron

        # init zonotopes
        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [-2., 1.], [2., 0.], [0., 1.]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators)
        # init intervals
        I1 = Interval(lb = np.array([1., -2.]), ub = np.array([3., 1.]))
        I2 = Interval(lb = np.array([-1., -2.]), ub = np.array([3., 1.]))
        # init vpolytopes
        VP1 = VPolytope(V = np.array([[1., -1.], [4., 0.], [-2., 2.]]))
        VP2 = VPolytope(V = np.array([[1., -1.], [4., 0.], [-2., 2.], [-4., -1.]]))
        # init hpolyhedra
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-5., 1.], [-5., -1.]]), b = np.array([1., 1., 1.]))

        # check containment
        assert Z1.contains(center)
        assert not Z1.contains(np.array([2., 0.]))
        assert Z2.contains(center)
        assert Z2.contains(np.array([3., 1.]))
        assert Z2.contains(np.array([4., 1.]))
        assert Z2.contains(Z1)
        assert Z2.contains(Z2)
        assert Z2.contains(I1)
        assert not Z2.contains(I2)
        assert Z2.contains(VP1)
        assert not Z2.contains(VP2)
        assert Z2.contains(HP1)
        assert not Z2.contains(HP2)


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

    def test_ConvexHull(self):
        # cases:
        # - interval x itself
        # - interval x np.ndarray (inside)
        # - interval x np.ndarray (boundary)
        # - interval x np.ndarray (outside)
        # - interval x interval (non-intersecting)
        # - interval x zonotope (mode = outer)
        # - interval x vpolytope
        # - interval x hpolyhedron

        # init intervals
        lower = np.array([-2., -1.])
        upper = np.array([3., 4.])
        I1 = Interval(lb = lower, ub = upper)
        v = np.array([-2., -1.])
        lower2 = np.array([8., 10.])
        upper2 = np.array([11., 15.])
        I2 = Interval(lb = lower2, ub = upper2)
        # init zonotope
        Z = Zonotope(c = np.array([6., 8.]), G = np.array([[1., -1.],[-1., 0.]]))
        # init vpolytope
        VP1 = VPolytope(V = np.array([[-1., 2.], [1., 0.], [1., 4.]]))
        VP2 = VPolytope(V = np.array([[-1., 0.], [1., -2.], [1., 2.]]))
        # init hpolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 3., -1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))

        # compute convex hull
        result1 = I1.convex_hull(I1)
        result2 = I1.convex_hull(I1.center())
        result3 = I1.convex_hull(lower)
        result4 = I1.convex_hull(lower + v, mode = 'outer')
        result5 = I1.convex_hull(I2, mode = 'outer')
        result6 = I1.convex_hull(Z, mode = 'outer')
        result7 = I1.convex_hull(VP1)
        result8 = I1.convex_hull(VP2, mode = 'outer')
        result9 = I1.convex_hull(HP1)
        result10 = I1.convex_hull(HP2, mode = 'outer')

        # manual computation
        true_result1 = I1
        true_result2 = I1
        true_result3 = I1
        true_result4 = Interval(lb = lower + v, ub = upper)
        true_result5 = Interval(lb = lower, ub = upper2)
        true_result6 = Interval(lb = lower, ub = np.array([8., 9.]))
        true_result7 = I1
        true_result8 = Interval(lb = np.array([-2., -2.]), ub = np.array([3., 4.]))
        true_result9 = I1
        true_result10 = true_result8

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
        assert result10 == true_result10

        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact convex hull with a point outside the interval
            I1.convex_hull(lower + v)
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact convex hull of two intervals where one is not contained in the other
            I1.convex_hull(I2)
        with self.assertRaises(NotImplementedError):
            # inner convex hull of two intervals not supported
            I1.convex_hull(I2, mode = 'inner')
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact convex hull with a zonotope that is not contained in the interval
            I1.convex_hull(Z)
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact convex hull with a vpolytope that is not contained in the interval
            I1.convex_hull(VP2)
        with self.assertRaises(ExactEvaluationImpossibleError):
            # exact convex hull with a hpolyhedron that is not contained in the interval
            I1.convex_hull(HP2)

        
        # cases:
        # - only center x only center
        # - only center x center and generators
        # - center and generators x only center
        # - center and generators x center and generators
        # - zonotope x interval
        # - zonotope x vpolytope
        # - zonotope x hpolyhedron

        # init zonotopes
        center1 = np.array([1., 0.])
        center2 = np.array([-1., 2.])
        generators1 = np.array([[2., -1.], [3., 2.], [1., 0.]])
        generators2 = np.array([[-1., 4.], [2., 0.], [0., 1.], [1., -1.], [3., 2.]])
        Z1 = Zonotope(c = center1)
        Z2 = Zonotope(c = center2)
        Z3 = Zonotope(c = center1, G = generators1)
        Z4 = Zonotope(c = center2, G = generators2)
        # init interval, vpolytope, hpolyhedron
        I = Interval(lb = center2, ub = center2)
        VP = VPolytope(V = np.array([[-2., 2.], [-3., 3.], [-4., 0.]]))
        HP = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([-2., 5., 3.]))

        # compute linear combination
        result1 = Z1.convex_hull(Z2, mode = 'outer')  # also 'exact'
        result2 = Z2.convex_hull(Z1, mode = 'outer')  # also 'exact'
        result3 = Z3.convex_hull(Z2, mode = 'outer')
        result4 = Z2.convex_hull(Z3, mode = 'outer')
        result5 = Z3.convex_hull(Z4, mode = 'outer')
        result6 = Z4.convex_hull(Z3, mode = 'outer')
        result7 = Z1.convex_hull(I, mode = 'outer')
        result8 = Z3.convex_hull(VP, mode = 'outer')
        result9 = Z3.convex_hull(HP, mode = 'outer')

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 1.]), G = np.array([[1., -1.]]))
        true_result2 = true_result1
        true_result3 = Zonotope(c = np.array([0., 1.]), G = np.array([[1., -1.], [2., -1.], [3., 2.], [1., 0.]]))
        true_result4 = Zonotope(c = np.array([0., 1.]), G = np.array([[-1., 1.], [2., -1.], [3., 2.], [1., 0.]]))
        true_result5 = Zonotope(c = np.array([0., 1.]),
                                G = np.array([[0.5, 1.5], [2.5, 1.], [0.5, 0.5], [-1., 1.], [-1.5, 2.5], [-0.5, -1.], [-0.5, 0.5], [1., -1.], [3., 2.]]))
        true_result6 = true_result5
        true_result7 = true_result1
        true_result8 = VPolytope(V = np.array([[-3., 3.], [-1., -3.], [1., -3.], [7., 1.], [3., 3.], [-5., -1.]]))
        true_result9 = VPolytope(V = np.array([[-4., 1.], [-2., 3.], [-1., -3.], [1., -3.], [7., 1.], [3., 3.], [-5., -1.]]))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5
        assert result6 == true_result6
        assert result7 == true_result7
        assert result8.contains(true_result8)
        assert result9.contains(true_result9)

        # check exceptions
        with self.assertRaises(NotImplementedError):
            Z1.convex_hull(Z2, mode = 'inner')  # should work
        with self.assertRaises(NotImplementedError):
            Z2.convex_hull(Z3, mode = 'exact')  # should work
        with self.assertRaises(NotImplementedError):
            Z2.convex_hull(VP)
        with self.assertRaises(NotImplementedError):
            Z2.convex_hull(HP)


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

    def test_Equals(self):
        ''' Test for set equality '''
        # cases:
        # interval x interval (itself)
        # interval x degenerate interval
        # interval x np.ndarray
        # single-point interval x np.ndarray (False)
        # single-point interval x np.ndarray (True)
        # intervals of different dimension
        # interval x zonotope
        # interval x vpolytope
        # interval x hpolyhedron

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
        Z1 = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 2.5]]))
        Z2 = Zonotope(c = np.array([0.5, 1.5]), G = np.array([[2.5, 0.],[0., 3.0]]))
        VP1 = VPolytope(V = np.array([[-2., -1.], [-2., 4.], [3., -1.], [3., 4.]]))
        VP2 = VPolytope(V = np.array([[-2., -1.], [-2., 4.], [3., -1.], [3., 4.01]]))
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                            b = np.array([3., 4., 2., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                            b = np.array([1., 1., 1., 1.01]))

        assert I1 == I1
        assert I1.__eq__(I1, rtol = 0., atol = 0.)
        assert not I1 == I2
        assert not I1 == lower
        assert not I3 == (lower + np.array([1., 0.]))
        assert I3 == lower
        assert not I1 == I4
        assert I1 == Z1
        assert not I1 == Z2
        assert I1 == VP1
        assert not I1 == VP2
        assert I1.__eq__(VP2, rtol = 0.1)
        assert I1 == HP1
        assert not I1 == HP2
        assert I1.__eq__(HP1, rtol = 0.1)


        # cases:
        # - only center
        # - center and all-zero generators
        # - center and generators
        # - center and -1*generators
        # - center and aligned generators
        # - zonotope x interval
        # - zonotope x vpolytope
        # - zonotope x hpolyhedron

        # init zonotopes
        center = np.array([1., 0.])
        center_3D = np.array([1., 0., 1.])
        generators_allzero = np.array([[0., 0.], [0., 0.], [0., 0.]])
        generators = np.array([[1., 2.], [2., 0.], [-1., 1.]])
        generators_reordered = np.array([[2., 0.], [1., 2.], [-1., 1.]])
        generators_neg = np.array([[2., 0.], [1., 2.], [1., -1.]])
        generators_aligned1 = np.array([[1., -1.], [2., 0.], [-1., 1.], [0., 1.], [2., 1.], [3., 1.5], [1., 0.]])
        generators_aligned2 = np.array([[-3., 0.], [0., 0.5], [0., -0.5], [2., -2.], [4., 2.], [-1., -0.5]])
        Z1 = Zonotope(c = center)
        Z2 = Zonotope(c = center, G = generators_allzero)
        Z3 = Zonotope(c = center, G = generators)
        Z4 = Zonotope(c = center, G = generators_reordered)
        Z5 = Zonotope(c = center, G = generators_neg)
        Z6 = Zonotope(c = center_3D)
        Z7 = Zonotope(c = center + np.array([1., 0.]))
        Z8 = Zonotope(c = center, G = generators_aligned1)
        Z9 = Zonotope(c = center, G = generators_aligned2)
        # init interval, vpolytope, hpolyhedron
        I = Interval(lb = center, ub = center)
        VP = VPolytope(V = np.array([[-1., -3.], [3., -3.], [5., 1.], [3., 3.], [-1., 3.], [-3., -1.]]))
        HP = HPolyhedron(A = np.array([[2./9., -1./9.], [0., -1./3.], [1./6., 1./6.], [-0.4, 0.2], [0., 1./3.], [-1./4., -1./4.]]),
                         b = np.array([1., 1., 1., 1., 1., 1.]))

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
        assert Z3 == VP
        assert Z3 == HP


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

    def test_Intersection(self):
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

    def test_Intersects(self):
        # cases:
        # - interval x vector
        # - interval x interval (itself)
        # - interval x interval (below)
        # - interval x interval (intersects from below)
        # - interval x interval (contained)
        # - interval x interval (intersects from above)
        # - interval x interval (above)
        # - interval x zonotope
        # - interval x vpolytope
        # - interval x hpolyhedron

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

        # init zonotopes
        Z1 = Zonotope(c = np.array([-2., 0., 1.]))
        Z2 = Zonotope(c = np.array([-4., 0., 1.]), G = np.array([[0., 1., 1.], [0., -1., 2.]]))
        # init vpolytopes
        VP1 = VPolytope(V = np.array([[-2., 0., 1.], [3., 1., 2.]]))
        VP2 = VPolytope(V = np.array([[-4., 0., 2.], [-5., 2., 0.], [-4., -2., -2.], [-4., 0., 4.]]))
        # init hpolyhedra
        HP1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                          b = np.array([1., 1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                          b = np.array([5., 1., 1., -3.]))
        
        # check intersection
        assert I.intersects(vector)
        assert not I.intersects(vector_outside)
        assert I.intersects(I)
        assert not I.intersects(I_below)
        assert I.intersects(I_intersects_below)
        assert I.intersects(I_contained)
        assert I.intersects(I_intersects_above)
        assert not I.intersects(I_above)
        
        assert I.intersects(Z1)
        assert not I.intersects(Z2)
        assert I.intersects(VP1)
        assert not I.intersects(VP2)
        assert I.intersects(HP1)
        assert not I.intersects(HP2)


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
        # - zonotope x vpolytope (single point)
        # - zonotope x vpolytope (full)
        # - zonotope x hpolyhedron

        # init sets
        center1 = np.array([1., 0.])
        generators1 = np.array([[1., -1.], [-1., 2.], [2., 1.], [0., 1.]])
        Z1 = Zonotope(c = center1, G = generators1)
        lower1 = np.array([2., 2.])
        upper1 = np.array([5., 3.])
        I1 = Interval(lb = lower1, ub = upper1)
        lower2 = np.array([-3., -5.])
        upper2 = np.array([-2., -2.])
        I2 = Interval(lb = lower2, ub = upper2)
        Z2 = Z1 + np.array([3., -2.])
        Z3 = Z1 + np.array([5., 5.])
        Z4 = Zonotope(c = center1)
        VP1 = VPolytope(V = center1)
        VP2 = VPolytope(V = np.array([[-3., -2.], [2., 0.], [-1., 3.]]))
        VP3 = VPolytope(V = np.array([[-4., -5.], [-2., -4.], [-6., 0.]]))
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([-2., 4., 4.]))
        HP2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([-2., 10., -2.]))

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
        assert Z1.intersects(VP1)
        assert Z1.intersects(VP2)
        assert not Z1.intersects(VP3)
        assert Z1.intersects(HP1)
        assert not Z1.intersects(HP2)


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

    def test_MinkowskiDifference(self):
        ''' Test for Minkowski difference '''
        # cases:
        # - interval - vector
        # - interval - interval
        # - interval - zonotope
        # todo interval - vpolytope
        # todo interval - hpolyhedron

        # init interval
        lower1 = np.array([-2., 3., 0.])
        upper1 = np.array([4., 9., 2.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([1., -2., 4.])
        lower2 = np.array([-2., 1., 0.])
        upper2 = np.array([1., 5., 1.])
        I2 = Interval(lb = lower2, ub = upper2)
        # init zonotope
        center = np.array([1., 0., 2.])
        generators = np.array([[1., 0., 0.], [-2., 1., 0.], [0., 1., 1.]])
        Z = Zonotope(c = center, G = generators)
        # init vpolytope
        VP1 = VPolytope(V = np.array([[-0.3, 0.1, 0.1], [0.1, -0.3, 0.1], [0.1, 0.1, -0.3], [0.1, 0.1, 0.1]]))
        VP2 = VPolytope(V = np.array([[-3., 1., 1.], [1., -3., 1.], [1., 1., -3.], [1., 1., 1.]]))
        # init hpolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                          b = np.array([0.1, 0.1, 0.1, 0.1]))
        HP2 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                          b = np.array([1., 1., 1., 1.]))

        # Minkowski difference
        result1 = I1.minkowski_difference(vector)
        result2 = I1.minkowski_difference(I2)
        result3 = I1.minkowski_difference(Z)
        result4 = I1.minkowski_difference(VP1)
        result5 = I1.minkowski_difference(HP1)

        # manual computation
        true_result1 = Interval(lb = np.array([-3., 5., -4.]),\
                                ub = np.array([3., 11., -2.]))
        true_result2 = Interval(lb = np.array([0., 2., 0.]),\
                                ub = np.array([3., 4., 1.]))
        true_result3 = Interval(lb = np.array([0., 5., -1.]),\
                                ub = np.array([0., 7., -1.]))
        true_result4 = Interval(lb = np.array([-1.7, 3.3, 0.3]),\
                                ub = np.array([3.9, 8.9, 1.9]))
        true_result5 = true_result4
        
        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

        # subtrahend too large -> empty set
        with self.assertRaises(EmptySetError):
            I2 = I1.matmul(2*np.eye(I1.dimension))
            I1.minkowski_difference(I2)
        with self.assertRaises(EmptySetError):
            I1.minkowski_difference(VP2)
        with self.assertRaises(EmptySetError):
            I1.minkowski_difference(HP2)


        # cases:
        # - zonotope - vector
        # - zonotope - zonotope
        # - zonotope - interval
        # - zonotope - vpolytope
        # - zonotope - hpolyhedron
        
        # init zonotope
        center = np.array([1., 0.])
        generators = np.array([[0., 1.], [1., 2.], [2., 1.], [-1., 0.]])
        Z = Zonotope(c = center, G = generators)

        # compute Minkowski difference
        result1 = Z.minkowski_difference(center)

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 0.]), G = generators)

        # check results
        assert result1 == true_result1

        # Minkowski difference between sets not implemented
        with self.assertRaises(NotImplementedError):
            Z.minkowski_difference(Z)
        with self.assertRaises(NotImplementedError):
            Z.minkowski_difference(Interval(lb = np.array([0., 0.]), ub = np.array([1., 2.])))
        with self.assertRaises(NotImplementedError):
            Z.minkowski_difference(VPolytope(V = np.array([[1., 0.], [0., 1.]])))
        with self.assertRaises(NotImplementedError):
            Z.minkowski_difference(HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                               b = np.array([1., 1., 1.])))
           
        
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

    def test_MinkowskiSum(self):
        # cases:
        # - interval x vector
        # - interval x interval
        # - interval x zonotope
        # - interval x vpolytope
        # - interval x hpolyhedron

        # init intervals
        lower1 = np.array([-2., 3., 0.])
        upper1 = np.array([4., 9., 2.])
        I1 = Interval(lb = lower1, ub = upper1)
        vector = np.array([1., -2., 4.])
        lower2 = np.array([-2., 1., 0.])
        upper2 = np.array([1., 5., 1.])
        I2 = Interval(lb = lower2, ub = upper2)
        # init zonotope
        center = np.array([1., 0., 2.])
        generators = np.array([[1., 0., 0.], [-2., 1., 0.], [0., 1., 1.]])
        Z = Zonotope(c = center, G = generators)
        # init vpolytope
        VP = VPolytope(V = np.array([[-3., 1., 1.], [1., -3., 1.], [1., 1., -3.], [1., 1., 1.]]))
        # init hpolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]]),
                         b = np.array([1., 1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([1., 0., 0.]), b = np.array([1.]))


        # Minkowski sum
        result1 = I1.minkowski_sum(vector)
        result2 = I1.minkowski_sum(I2)
        result3 = I1.minkowski_sum(Z, mode = 'outer')
        result4 = I1.minkowski_sum(VP, mode = 'outer')
        result5 = I1.minkowski_sum(HP1, mode = 'outer')

        # manual computation
        true_result1 = Interval(lb = np.array([-1., 1., 4.]),\
                                ub = np.array([5., 7., 6.]))
        true_result2 = Interval(lb = np.array([-4., 4., 0.]),\
                                ub = np.array([5., 14., 3.]))
        true_result3 = Interval(lb = np.array([-4., 1., 1.]),\
                                ub = np.array([8., 11., 5.]))
        true_result4 = Interval(lb = np.array([-5., 0., -3.]),\
                                ub = np.array([5., 10., 3.]))
        true_result5 = true_result4

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5 == true_result5

        with self.assertRaises(ExactEvaluationImpossibleError):
            I1.minkowski_sum(Z)
        with self.assertRaises(ExactEvaluationImpossibleError):
            I1.minkowski_sum(VP)
        with self.assertRaises(ExactEvaluationImpossibleError):
            I1.minkowski_sum(HP1)
        with self.assertRaises(UnboundedSetError):
            I1.minkowski_sum(HP2, mode = 'outer')


        # cases:
        # - zonotope + zonotope
        # - zonotope + vector
        # - zonotope + interval
        # - zonotope + vpolytope
        # - zonotope + hpolyhedron

        # init zonotopes
        center1 = np.array([1., 0.])
        generators1 = np.array([[1., -1.], [2., 0.], [2., -1.]])
        center2 = np.array([-1., 1.])
        generators2 = np.array([[0., 1.], [-1., -1.], [3., 0.]])
        Z1 = Zonotope(c = center1, G = generators1)
        Z2 = Zonotope(c = center2, G = generators2)
        v = np.array([-2., 0.])
        Z3 = Zonotope(c = v)
        # init interval, vpolytope, hpolyhedron
        I = Interval(lb = v, ub = v)
        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))
        VP2 = VPolytope(V = np.array([[0., -1.], [2., -1.], [6., 1.], [4., 3.], [2., 3.], [-2., 1.]]))
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                         b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[0., -1.], [-1., -1.], [0.25, -0.5], [0., 1./3.], [1./7., 1./7.], [-0.25, 0.5]]),
                          b = np.ones(6))

        # compute Minkowski sums
        result1 = Z1.minkowski_sum(Z2)
        result2 = Z1.minkowski_sum(v)
        result3 = Z1.minkowski_sum(I)
        result4 = Z3.minkowski_sum(Z1)
        result5 = Z1.minkowski_sum(VP1, mode = 'outer')
        result6 = Z1.minkowski_sum(HP1, mode = 'outer')

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 1.]),\
                                G = np.array([[1., -1.], [2., 0.], [2., -1.], [0., 1.], [-1., -1.], [3., 0.]]))
        true_result2 = Zonotope(c = np.array([-1., 0.]), G = generators1)
        true_result3 = true_result2
        true_result4 = true_result2
        true_result5 = HPolyhedron(A = np.array([[1./6., 1./3.], [1./7., 1./7.], [0., 1./3.], [0., -1./3.],
                                                 [1./11., -2./11.], [1./13., -2./13.], [-2./15., 1./15.],
                                                 [-0.2, -0.4], [1./23., -2./23.], [-2./23., 1./23.], [-2./11., 1./11.],
                                                 [-0.25, -0.25], [0.2, 0.2]]),
                                   b = np.ones(13))
        true_result6 = HPolyhedron(A = np.array([[-1./9., 1./9.], [-0.2, -0.2], [-1./15, 1./15.], [-1./7., 1./7.],
                                                 [1./7., 1./7.], [1./9., 2./9.], [0., 0.25], [1./7., 0.],
                                                 [1./9., 0.], [0., -0.25], [1./11., 0.], [-0.2, -0.4], [-1./3., -1./3.]]),
                                   b = np.ones(13))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4
        assert result5.contains(true_result5)
        assert result6.contains(true_result6)

        # exact evaluations currently not implemented, sometimes impossible
        with self.assertRaises(ExactEvaluationImpossibleError):
            Z1.minkowski_sum(VP1)
        with self.assertRaises(NotImplementedError):
            Z1.minkowski_sum(VP2)
        with self.assertRaises(ExactEvaluationImpossibleError):
            Z1.minkowski_sum(HP1)
        with self.assertRaises(NotImplementedError):
            Z1.minkowski_sum(HP2)


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