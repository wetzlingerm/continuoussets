
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
from continuoussets.utils.exceptions import EmptySetError, UnboundedSetError, ExactEvaluationImpossibleError, OtherFunctionError
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
        result1 = cs.cartesian_product(I1, I1)
        result2 = cs.cartesian_product(I1, lower)
        result3 = cs.cartesian_product(I1, Z1)
        result4 = cs.cartesian_product(I1, Z2, mode = 'outer')
        result5 = cs.cartesian_product(I1, VP1)
        result6 = cs.cartesian_product(I1, VP2)
        result7 = cs.cartesian_product(I1, HP1)
        result8 = cs.cartesian_product(I1, HP2, mode = 'outer')

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
        
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)
        assert cs.equals(result6, true_result6)
        assert cs.equals(result7, true_result7)
        assert cs.equals(result8, true_result8)


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
        result1 = cs.cartesian_product(Z1_onlycenter, Z2_onlycenter)
        result2 = cs.cartesian_product(Z1_onlycenter, center2)
        result3 = cs.cartesian_product(Z1_onlycenter, Z4)
        result4 = cs.cartesian_product(Z3, center2)
        result5 = cs.cartesian_product(Z3, Z2_onlycenter)
        result6 = cs.cartesian_product(Z3, Z4)
        result7 = cs.cartesian_product(Z1_onlycenter, I)
        result8 = cs.cartesian_product(Z3, VP1, mode = 'outer')
        result9 = cs.cartesian_product(Z3, VP2)
        result10 = cs.cartesian_product(Z3, HP, mode = 'outer')
        
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
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)
        assert cs.equals(result6, true_result6)
        assert cs.equals(result7, true_result7)
        assert cs.equals(result8, true_result8)
        assert cs.equals(result9, true_result9)
        assert cs.equals(result10, true_result10)

    
        # cases:
        # - HPolytope x vector
        # - HPolytope x HPolytope
        # - HPolytope x Interval
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.]]), b = np.array([1., 1.]))
        HP2 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([3., 2.]))
        I = Interval(lb = -2., ub = 3.)

        result1 = cs.cartesian_product(HP1, np.array([2.]))
        result2 = cs.cartesian_product(HP1, HP2)
        result3 = cs.cartesian_product(HP1, I)

        true_result1 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
                                   b = np.array([1., 1., 2., -2.]))
        true_result2 = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
                                   b = np.array([1., 1., 3., 2.]))
        
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result2)


        # cases:
        # - vpolytope x vpolytope
        # - vpolytope x interval
        VP1 = VPolytope(V = np.array([[1.], [2.], [4.]]))
        VP2 = VPolytope(V = np.array([[-3.], [5.]]))
        I2 = Interval(lb = np.array([-3.]), ub = np.array([5.]))

        V1_V2 = np.array([[1., -3.], [1., 5.], [4., -3.], [4., 5.]])

        VP1_VP2 = cs.cartesian_product(VP1, VP2)
        VP1_VP2 = VP1_VP2.compact()
        VP1_I2 = cs.cartesian_product(VP1, I2)
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
        assert cs.contains(I1, I1)
        assert cs.contains(I1, I2)
        assert not cs.contains(I1, I3)
        assert not cs.contains(I1, I4)
        assert cs.contains(I1, lower + np.array([1., 0.]))
        assert cs.contains(I1, lower)
        assert not cs.contains(I1, lower + np.array([-1., 0.]))
        assert cs.contains(I1, Z1)
        assert cs.contains(I1, Z2)
        assert not cs.contains(I1, Z3)
        assert not cs.contains(I1, Z4)
        assert cs.contains(I1, VP1)
        assert not cs.contains(I1, VP2)
        assert cs.contains(I1, HP1)
        assert not cs.contains(I1, HP2)


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
        assert cs.contains(Z1, center)
        assert not cs.contains(Z1, np.array([2., 0.]))
        assert cs.contains(Z2, center)
        assert cs.contains(Z2, np.array([3., 1.]))
        assert cs.contains(Z2, np.array([4., 1.]))
        assert cs.contains(Z2, Z1)
        assert cs.contains(Z2, Z2)
        assert cs.contains(Z2, I1)
        assert not cs.contains(Z2, I2)
        assert cs.contains(Z2, VP1)
        assert not cs.contains(Z2, VP2)
        assert cs.contains(Z2, HP1)
        assert not cs.contains(Z2, HP2)


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

        assert cs.contains(HP1, v_inside)
        assert cs.contains(HP1, v_boundary)
        assert cs.contains(HP1, HP1)
        assert cs.contains(HP2, HP1)
        assert not cs.contains(HP1, HP2)


        # cases:
        # - vpolytope x vector
        # - vpolytope x vpolytope (self)
        # - vpolytope x interval
        V_2D = np.array([[1., -1.], [-2., 0.], [0., 1.]])
        VP_2D = VPolytope(V = V_2D)
        v = np.array([0., 0.])
        I_2D = Interval(lb = np.array([-0.1, -0.1]), ub = np.array([0.1, 0.1]))

        assert cs.contains(VP_2D, v)
        assert cs.contains(VP_2D, VP_2D)
        assert cs.contains(VP_2D, I_2D)

    def test_ConvexHull(self):
        # cases:
        # - interval
        # - zonotope
        # - vpolytope
        # - hpolyhedron

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
        result1 = cs.convex_hull(I1, I1)
        result2 = cs.convex_hull(I1, I1.center())
        result3 = cs.convex_hull(I1, lower)
        result4 = cs.convex_hull(I1, lower + v, mode = 'outer')
        result5 = cs.convex_hull(I1, I2, mode = 'outer')
        result6 = cs.convex_hull(I1, Z, mode = 'outer')
        result7 = cs.convex_hull(I1, VP1)
        result8 = cs.convex_hull(I1, VP2, mode = 'outer')
        result9 = cs.convex_hull(I1, HP1)
        result10 = cs.convex_hull(I1, HP2, mode = 'outer')

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
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)
        # assert cs.equals(result6, true_result6)
        assert cs.equals(result7, true_result7)
        # assert cs.equals(result8, true_result8)
        assert cs.equals(result9, true_result9)
        # assert cs.equals(result10, true_result10)

        with self.assertRaises(NotImplementedError):
            # inner convex hull of two intervals not supported
            cs.convex_hull(I1, I2, mode = 'inner')

        
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
        result1 = cs.convex_hull(Z1, Z2, mode = 'outer')  # also 'exact'
        result2 = cs.convex_hull(Z2, Z1, mode = 'outer')  # also 'exact'
        result3 = cs.convex_hull(Z3, Z2, mode = 'outer')
        result4 = cs.convex_hull(Z2, Z3, mode = 'outer')
        result5 = cs.convex_hull(Z3, Z4, mode = 'outer')
        result6 = cs.convex_hull(Z4, Z3, mode = 'outer')
        result7 = cs.convex_hull(Z1, I, mode = 'outer')
        result8 = cs.convex_hull(Z3, VP, mode = 'outer')
        result9 = cs.convex_hull(Z3, HP, mode = 'outer')

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
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)
        assert cs.equals(result6, true_result6)
        assert cs.equals(result7, true_result7)
        assert cs.contains(result8, true_result8)
        assert cs.contains(result9, true_result9)


        # cases:
        # - HPolyhedron x HPolyhedron
        # - HPolyhedron x vector
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([0., 0., 1.]))
        HP2 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.], [1., 1.]]), b = np.array([0., 0., 1.]))
        v = np.array([2., 1.])

        result1 = cs.convex_hull(HP1, HP2, mode = 'outer')
        result2 = cs.convex_hull(HP2, HP1, mode = 'outer')
        result3 = cs.convex_hull(HP1, v, mode = 'outer')

        true_result1 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                                   b = np.array([1., 1., 1., 1.]))
        true_result3 = HPolyhedron(A = np.array([[1., -1.], [-1., 3.], [-1., -1.]]),
                                   b = np.array([1., 1., 1.]))
        
        assert cs.contains(result1, HP1)
        assert cs.contains(result1, HP2)
        assert cs.contains(result2, HP1)
        assert cs.contains(result2, HP2)
        assert cs.contains(result1, true_result1)
        assert cs.contains(result2, true_result1)
        assert cs.contains(result3, true_result3)


        # cases:
        # - single vertex x single vertex
        # - vpolytope x vpolytope
        # - vpolytope x zonotope
        VP1 = VPolytope(V = np.array([1., 1.]))
        VP2 = VPolytope(V = np.array([0., 1.]))
        VP3 = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., -1.]]))
        VP4 = VPolytope(V = np.array([[1., 0.], [0., 0.], [0., 1.]]))
        Z = Zonotope(c = [1., 1.])

        result1 = cs.convex_hull(VP1, VP2)
        result2 = cs.convex_hull(VP3, VP4)
        result3 = cs.convex_hull(VP3, Z)

        true_result1 = VPolytope(V = np.array([[1., 1.], [0., 1.]]))
        true_result2 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
        true_result3 = VPolytope(V = np.array([[1., 1.], [-1., 0.], [0., -1.]]))

        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)

    def test_Equals(self):
        ''' Test for set equality '''
        # cases:
        # - single point
        # - 1D
        # - interval
        # - zonotope
        # - polytope
        point = np.array([3., 2., -1.])
        I_point = Interval(lb = point)
        Z_point = Zonotope(c = point)
        VP_point = VPolytope(V = point)
        HP_point = HPolyhedron(A = np.array([[1., 0., 0.], [0., 1. ,0.], [0., 0., 1.], [-1., -1., -1.]]),
                               b = np.array([3., 2., -1., -4.]))

        Z_1D = Zonotope(c = np.array([2.5]),
                        G = np.array([[1.], [-0.5]]))
        I_1D = Interval(lb = np.array([1.]), ub = np.array([4.]))
        VP_1D = VPolytope(V = np.array([[1.], [2.], [4.], [3.5]]))
        HP_1D = HPolyhedron(A = np.array([[1.], [-1.], [1.]]),
                            b = np.array([4., -1., 7.]))

        I_interval = Interval(lb = np.array([-1., 0.]), ub = np.array([3., 2.]))
        Z_interval = Zonotope(c = np.array([1., 1.]), G = np.array([[2., 0.], [0., 0.5], [0., -0.5]]))
        VP_interval = VPolytope(V = np.array([[-1., 0.], [3., 0.], [0., 0.], [3., 2.], [-1., 2.]]))
        HP_interval = HPolyhedron(A = np.array([[1., 0.], [0., 0.5], [1., 1.], [0., -2.], [-1., 0.]]),
                                  b = np.array([3., 1., 10., 0., 1.]))
        
        Z_zonotope = Zonotope(c = np.array([1., 0.]), G = np.array([[0.5, 1.], [0., -1.], [1., 2.]]))
        VP_zonotope = VPolytope(V = np.array([[-0.5, -4.], [0.5, -2.], [2.5, 2.], [2.5, 4.], [1.5, 2.], [-0.5, -2.]]))
        HP_zonotope = HPolyhedron(A = np.array([[2./3., -1./3.], [-2., 0.], [2./3., -1./3.], [-2., 1.], [0.4, 0.], [-2., 1.]]),
                                  b = np.array([1., 1., 1., -1., 1., -1.]))
        
        VP_polytope = VPolytope(V = np.array([[-1., 0.], [1., -2.], [1., 2.]]))
        HP_polytope = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.array([1., 1., 1.]))

        Z_deg = Zonotope(c = np.array([2., -1., 1.]),
                         G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.], [4., 2., -2.], [-1., 4., 5.]]))
        VP_deg = VPolytope(V = np.array([[-1., -13., -8.], [-3., -11., -4.], [9., -5., -10.], [7., -9., -12.],
                                         [9., 1., -4.], [-5., -3., 6.], [-3., 7., 14.], [-5., 3., 12.], [7., 9., 6.], [5., 11., 10.]]))
        HP_deg = HPolyhedron(A = np.array([[-1./np.sqrt(2.), 0., 1./np.sqrt(2.)],
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
        
        HP_unb = HPolyhedron(A = np.array([[1., 0., 0.]]), b = np.array([1.]))

        # point checks
        assert cs.equals(I_point, I_point)
        assert cs.equals(I_point, Z_point)
        assert cs.equals(I_point, VP_point)
        assert cs.equals(I_point, HP_point)
        assert cs.equals(Z_point, I_point)
        assert cs.equals(Z_point, Z_point)
        assert cs.equals(Z_point, VP_point)
        assert cs.equals(Z_point, HP_point)
        assert cs.equals(VP_point, I_point)
        assert cs.equals(VP_point, Z_point)
        assert cs.equals(VP_point, VP_point)
        assert cs.equals(VP_point, HP_point)
        assert cs.equals(HP_point, I_point)
        assert cs.equals(HP_point, Z_point)
        assert cs.equals(HP_point, VP_point)
        assert cs.equals(HP_point, HP_point)

        # 1D checks
        assert cs.equals(I_1D, I_1D)
        assert cs.equals(I_1D, Z_1D)
        assert cs.equals(I_1D, VP_1D)
        assert cs.equals(I_1D, HP_1D)
        assert cs.equals(Z_1D, I_1D)
        assert cs.equals(Z_1D, Z_1D)
        assert cs.equals(Z_1D, VP_1D)
        assert cs.equals(Z_1D, HP_1D)
        assert cs.equals(VP_1D, I_1D)
        assert cs.equals(VP_1D, Z_1D)
        assert cs.equals(VP_1D, VP_1D)
        assert cs.equals(VP_1D, HP_1D)
        assert cs.equals(HP_1D, I_1D)
        assert cs.equals(HP_1D, Z_1D)
        assert cs.equals(HP_1D, VP_1D)
        assert cs.equals(HP_1D, HP_1D)

        # interval checks
        assert cs.equals(I_interval, I_interval)
        assert cs.equals(I_interval, Z_interval)
        assert cs.equals(I_interval, VP_interval)
        assert cs.equals(I_interval, HP_interval)
        assert cs.equals(Z_interval, I_interval)
        assert cs.equals(Z_interval, Z_interval)
        assert cs.equals(Z_interval, VP_interval)
        assert cs.equals(Z_interval, HP_interval)
        assert cs.equals(VP_interval, I_interval)
        assert cs.equals(VP_interval, Z_interval)
        assert cs.equals(VP_interval, VP_interval)
        assert cs.equals(VP_interval, HP_interval)
        assert cs.equals(HP_interval, I_interval)
        assert cs.equals(HP_interval, Z_interval)
        assert cs.equals(HP_interval, VP_interval)
        assert cs.equals(HP_interval, HP_interval)

        # zonotope checks
        assert cs.equals(Z_zonotope, Z_zonotope)
        assert cs.equals(Z_zonotope, VP_zonotope)
        assert cs.equals(Z_zonotope, HP_zonotope)
        assert cs.equals(VP_zonotope, Z_zonotope)
        assert cs.equals(VP_zonotope, VP_zonotope)
        assert cs.equals(VP_zonotope, HP_zonotope)
        assert cs.equals(HP_zonotope, Z_zonotope)
        assert cs.equals(HP_zonotope, VP_zonotope)
        assert cs.equals(HP_zonotope, HP_zonotope)

        # degenerate set checks
        assert cs.equals(Z_deg, Z_deg)
        assert cs.equals(Z_deg, VP_deg)
        assert cs.equals(Z_deg, HP_deg, rtol = 1e-5)
        assert cs.equals(VP_deg, Z_deg)
        assert cs.equals(VP_deg, VP_deg)
        assert cs.equals(VP_deg, HP_deg, rtol = 1e-5)
        assert cs.equals(HP_deg, Z_deg, rtol = 1e-5)
        assert cs.equals(HP_deg, VP_deg, rtol = 1e-5)
        assert cs.equals(HP_deg, HP_deg)

        # unbounded
        assert cs.equals(HP_unb, HP_unb)

        # todo: tests that should return False

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

        result1 = cs.intersection(HP1, np.array([0., 0.]))
        result2 = cs.intersection(HP1, HP2)
        result3 = cs.intersection(HP2, HP1)
        result4 = cs.intersection(HP1, I)

        true_result2 = HPolyhedron(A = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]),
                                   b = np.array([1., 1., 1., 1.]))
        true_result4 = HPolyhedron(A = np.array([[1., 0.], [0., -1.], [-1., 1.]]),
                                   b = np.array([0., 0., 1.]))

        assert cs.equals(result1, np.array([0., 0.]))
        assert cs.equals(result2, true_result2)
        assert cs.equals(result2, result3)
        assert cs.equals(result4, true_result4)
        
        with self.assertRaises(EmptySetError):
            cs.intersection(HP1, np.array([10., 5.]))

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
        assert cs.intersects(I, vector)
        assert not cs.intersects(I, vector_outside)
        assert cs.intersects(I, I)
        assert not cs.intersects(I, I_below)
        assert cs.intersects(I, I_intersects_below)
        assert cs.intersects(I, I_contained)
        assert cs.intersects(I, I_intersects_above)
        assert not cs.intersects(I, I_above)
        
        assert cs.intersects(I, Z1)
        assert not cs.intersects(I, Z2)
        assert cs.intersects(I, VP1)
        assert not cs.intersects(I, VP2)
        assert cs.intersects(I, HP1)
        assert not cs.intersects(I, HP2)


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
        assert cs.intersects(Z1, center1)
        assert cs.intersects(Z1, np.array([3., 3.]))
        assert not cs.intersects(Z1, np.array([4., 2.]))
        assert cs.intersects(Z1, I1)
        assert not cs.intersects(Z1, I2)
        assert cs.intersects(Z1, Z2)
        assert not cs.intersects(Z1, Z3)
        assert cs.intersects(Z1, Z4)
        assert cs.intersects(Z4, Z1)
        assert cs.intersects(Z1, VP1)
        assert cs.intersects(Z1, VP2)
        assert not cs.intersects(Z1, VP3)
        assert cs.intersects(Z1, HP1)
        assert not cs.intersects(Z1, HP2)


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

        assert cs.intersects(HP1, np.array([0., 0.]))
        assert not cs.intersects(HP1, np.array([10., 5.]))
        assert cs.intersects(HP1, HP2)
        assert cs.intersects(HP2, HP1)
        assert cs.intersects(HP1, I1)
        assert not cs.intersects(HP1, I2)
        assert cs.intersects(HP1, Z1)
        assert not cs.intersects(HP1, Z2)
        assert cs.intersects(HP1, VP1)
        assert not cs.intersects(HP1, VP2)


        # cases:
        # - vpolytope x vector
        # - vpolytope x vpolytope (self)
        # - vpolytope x vpolytope
        # - vpolytope x single vertex
        # - vpolytope x interval
        # - vpolytope x zonotope
        # - vpolytope x hpolyhedron
        VP1 = VPolytope(V = np.array([[-1., 0.], [1., 1.], [0., -1.]]))
        VP2 = VPolytope(V = np.array([[1., 0.], [0., 0.], [0., 1.]]))
        VP3 = VPolytope(V = np.array([0.25, 0.25]))
        I1 = Interval(lb = [-1., 0.], ub = [0., 1.])
        I2 = Interval(lb = [0.75, -1.], ub = [1., 0.])
        Z_1 = Zonotope(c = np.array([1., -1.]), G = np.array([[1., -1.], [0.5, 0.]]))
        Z_2 = Zonotope(c = np.array([1., -1.]), G = np.array([[1., 1.], [0.5, 0.]]))
        HP_1 = HPolyhedron(A = np.array([[0., 1.]]), b = np.array([-0.5]))
        HP_2 = HPolyhedron(A = np.array([[-1., 1.]]), b = np.array([-1.1]))

        assert cs.intersects(VP1, np.array([0., 0.]))
        assert cs.intersects(VP1, np.array([1., 1.]))
        assert cs.intersects(VP1, VP1)
        assert cs.intersects(VP1, VP2)
        assert cs.intersects(VP2, VP1)
        assert cs.intersects(VP1, VP3)
        assert cs.intersects(VP3, VP1)
        assert cs.intersects(VP1, I1)
        assert not cs.intersects(VP1, I2)
        assert cs.intersects(VP1, Z_1)
        assert not cs.intersects(VP1, Z_2)
        assert cs.intersects(VP1, HP_1)
        assert not cs.intersects(VP1, HP_2)

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
        result1 = cs.minkowski_difference(I1, vector)
        result2 = cs.minkowski_difference(I1, I2)
        result3 = cs.minkowski_difference(I1, Z)
        result4 = cs.minkowski_difference(I1, VP1)
        result5 = cs.minkowski_difference(I1, HP1)

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
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)
        assert cs.equals(result5, true_result5)

        # subtrahend too large -> empty set
        with self.assertRaises(EmptySetError):
            I2 = I1.matmul(2*np.eye(I1.dimension))
            cs.minkowski_difference(I1, I2)
        with self.assertRaises(EmptySetError):
            cs.minkowski_difference(I1, VP2)
        with self.assertRaises(EmptySetError):
            cs.minkowski_difference(I1, HP2)


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
        result1 = cs.minkowski_difference(Z, center)

        # manual computation
        true_result1 = Zonotope(c = np.array([0., 0.]), G = generators)

        # check results
        assert cs.equals(result1, true_result1)

        # Minkowski difference between sets not implemented
        with self.assertRaises(NotImplementedError):
            cs.minkowski_difference(Z, Z)
        with self.assertRaises(NotImplementedError):
            cs.minkowski_difference(Z, Interval(lb = np.array([0., 0.]), ub = np.array([1., 2.])))
        with self.assertRaises(NotImplementedError):
            cs.minkowski_difference(Z, VPolytope(V = np.array([[1., 0.], [0., 1.]])))
        with self.assertRaises(NotImplementedError):
            cs.minkowski_difference(Z, HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                               b = np.array([1., 1., 1.])))
           

        center = np.array([1., 0.])
        generators = np.array([[1., -1.], [2., 0.], [2., -1.]])
        Z1 = Zonotope(c = center, G = generators)
        # call minkowski_difference insetead of __sub__ with two IConvexSet objects
        with self.assertRaises(OtherFunctionError):
            Z1 - Z1
        with self.assertRaises(OtherFunctionError):
            Z1 - Interval(lb = np.array([0., 1.]), ub = np.array([2., 4.]))
        with self.assertRaises(OtherFunctionError):
            Z1 - VPolytope(V = np.array([[1., 0.], [0., 1.]]))
        with self.assertRaises(OtherFunctionError):
            Z1 - HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        
        # cases:
        # - hpolyhedron - interval
        # - hpolyhedron - vector
        HP = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                         b = np.array([1., 1., 1.]))
        I = Interval(lb = np.array([-0.1, -0.2]), ub = np.array([0.2, 0.3]))
        v = np.array([2., -1.])

        result1 = cs.minkowski_difference(HP, I)
        result2 = cs.minkowski_difference(HP, v)

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([0.8, 0.6, 0.7]))
        true_result2 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                   b = np.array([-1., 4., 2.]))
        
        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result2)

        with self.assertRaises(OtherFunctionError):
            HP1 - HP1


        # cases:
        # - vpolytope - vector
        # - vpolytope - vpolytope (vector)
        # - vpolytope - vpolytope
        VP1 = VPolytope(V = np.array([[2., 0.], [-1., -1.], [-2., 1.]]))
        VP2 = VPolytope(V = np.array([1., -1.]))
        VP3 = VPolytope(V = np.array([[2., 0.], [-1., 1.]]))

        result1 = cs.minkowski_difference(VP1, np.array([1., -1.]))
        result2 = cs.minkowski_difference(VP1, VP2)

        true_result1 = VPolytope(V = np.array([[1., 1.], [-2., 0.], [-3., 2.]]))

        assert cs.equals(result1, true_result1)
        assert cs.equals(result2, true_result1)
        with self.assertRaises(NotImplementedError):
            cs.minkowski_difference(VP1, VP3)


        # check exceptions
        with self.assertRaises(OtherFunctionError):
            # call minkowski_difference instead of __sub__
            VP1 - VP1

    def test_MinkowskiSum(self):
        # cases:
        # - interval
        # - zonotope
        # - vpolytope
        # - hpolyhedron

        I1 = Interval(lb = np.array([-1., 0.]), ub = np.array([3., 1.]))
        I2 = Interval(lb = np.array([-2., 3.]), ub = np.array([4., 9.]))

        Z1 = Zonotope(c = np.array([1., 0.]), G = np.array([-1., 1.]))
        Z2 = Zonotope(c = np.array([1., -1.]), G = np.array([[0., 1.], [-1., -1.], [3., 0.]]))

        VP1 = VPolytope(V = np.array([[1., 0.], [0., 1.], [-1., -1.]]))
        VP2 = VPolytope(V = np.array([[0., -1.], [2., -1.], [6., 1.], [4., 3.], [2., 3.], [-2., 1.]]))

        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                          b = np.array([1., 1., 1.]))
        HP2 = HPolyhedron(A = np.array([[0., 1.], [-1., -1.], [1., -1.]]),
                          b = np.array([1., 1., 1.]))

        # Minkowski sums
        sum_interval_interval = cs.minkowski_sum(I1, I2)
        sum_interval_interval_ = cs.minkowski_sum(I2, I1)

        sum_interval_zonotope = cs.minkowski_sum(I1, Z1)
        sum_interval_zonotope_ = cs.minkowski_sum(Z1, I1)

        sum_interval_vpolytope = cs.minkowski_sum(I1, VP1)
        sum_interval_vpolytope_ = cs.minkowski_sum(VP1, I1)

        sum_interval_hpolyhedron = cs.minkowski_sum(I1, HP1)
        sum_interval_hpolyhedron_ = cs.minkowski_sum(HP1, I1)

        sum_zonotope_zonotope = cs.minkowski_sum(Z1, Z2)
        sum_zonotope_zonotope_ = cs.minkowski_sum(Z2, Z1)

        sum_zonotope_vpolytope = cs.minkowski_sum(Z1, VP1)
        sum_zonotope_vpolytope_ = cs.minkowski_sum(VP1, Z1)

        sum_zonotope_hpolyhedron = cs.minkowski_sum(Z1, HP1)
        sum_zonotope_hpolyhedron_ = cs.minkowski_sum(HP1, Z1)

        sum_vpolytope_vpolytope = cs.minkowski_sum(VP1, VP2)
        sum_vpolytope_vpolytope_ = cs.minkowski_sum(VP1, VP2)

        sum_vpolytope_hpolyhedron = cs.minkowski_sum(VP1, HP1)
        sum_vpolytope_hpolyhedron_ = cs.minkowski_sum(VP1, HP1)

        sum_hpolyhedron_hpolyhedron = cs.minkowski_sum(HP1, HP2)
        sum_hpolyhedron_hpolyhedron_ = cs.minkowski_sum(HP2, HP1)

        # manual computations
        true_sum_interval_interval = Interval(lb = np.array([-3., 3.]), ub = np.array([7., 10.]))
        true_sum_interval_zonotope = Zonotope(c = np.array([2., 0.5]), G = np.array([[-1., 1.], [2., 0.], [0., 0.5]]))
        true_sum_interval_vpolytope = VPolytope(V = np.array([[-2., -1.], [-2., 0.], [-1., 2.], [2., -1.], [3., 2.], [4., 0.], [4., 1.]]))
        true_sum_interval_hpolyhedron = HPolyhedron(A = np.array([[0., 1./3.], [0., -0.5], [0.25, 0.], [-1./3., 1./3.], [-0.5, 0.], [-0.5, -0.5]]),
                                                    b = np.array([1., 1., 1., 1., 1., 1.]))
        true_sum_zonotope_zonotope = Zonotope(c = np.array([2., -1.]),
                                              G = np.array([[-1., 1.], [0., 1.], [-1., -1.], [3., 0.]]))
        true_sum_zonotope_vpolytope = HPolyhedron(A = np.array([[0.25, -0.25], [-0.5, 0.5], [0.2, -0.4], [-1., -1.], [-1., 0.5], [0.5, 0.5]]),
                                                  b = np.array([1., 1., 1., 1., 1., 1.]))
        true_sum_zonotope_hpolyhedron = HPolyhedron(A = np.array([[1., 0.], [1./np.sqrt(2.), -1./np.sqrt(2.)], [1./np.sqrt(2.), 1./np.sqrt(2.)], [-1./np.sqrt(2.), 1./np.sqrt(2.)], [-1./np.sqrt(2.), -1./np.sqrt(2.)]]),
                                                    b = np.array([3., 3.*np.sqrt(3.), 2.*np.sqrt(2.), np.sqrt(2.), 0.]))
        true_sum_vpolytope_vpolytope = VPolytope(V = np.array([[-1., -2.], [1., -2.], [7., 1.], [4., 4.], [2., 4.], [-2., 2.], [-3., 0.]]))
        true_sum_vpolytope_hpolyhedron = HPolyhedron(A = np.array([[0.5, 0.], [-0.5, 0.5], [0.25, 0.25], [-1./3., -1./3.], [1./6., -1./3.], [-2./3., 1./3.]]),
                                                     b = np.array([1., 1., 1., 1., 1., 1.]))
        true_sum_hpolyhedron_hpolyhedron = HPolyhedron(A = np.array([[-0.25, 0.25], [1./3., 0.], [0.25, -0.25], [0., 1./3.], [-0.5, -0.5]]),
                                                       b = np.array([1., 1., 1., 1., 1.]))

        # check results
        assert cs.equals(sum_interval_interval, true_sum_interval_interval)
        assert cs.equals(sum_interval_interval_, true_sum_interval_interval)

        assert cs.equals(sum_interval_zonotope, true_sum_interval_zonotope)
        assert cs.equals(sum_interval_zonotope_, true_sum_interval_zonotope)

        assert cs.equals(sum_interval_vpolytope, true_sum_interval_vpolytope)
        assert cs.equals(sum_interval_vpolytope_, true_sum_interval_vpolytope)

        assert cs.equals(sum_interval_hpolyhedron, true_sum_interval_hpolyhedron)
        assert cs.equals(sum_interval_hpolyhedron_, true_sum_interval_hpolyhedron)

        assert cs.equals(sum_zonotope_zonotope, true_sum_zonotope_zonotope)
        assert cs.equals(sum_zonotope_zonotope_, true_sum_zonotope_zonotope)

        assert cs.equals(sum_zonotope_vpolytope, true_sum_zonotope_vpolytope)
        assert cs.equals(sum_zonotope_vpolytope_, true_sum_zonotope_vpolytope)

        assert cs.equals(sum_zonotope_hpolyhedron, true_sum_zonotope_hpolyhedron)
        assert cs.equals(sum_zonotope_hpolyhedron_, true_sum_zonotope_hpolyhedron)

        assert cs.equals(sum_vpolytope_vpolytope, true_sum_vpolytope_vpolytope)
        assert cs.equals(sum_vpolytope_vpolytope_, true_sum_vpolytope_vpolytope)

        assert cs.equals(sum_vpolytope_hpolyhedron, true_sum_vpolytope_hpolyhedron)
        assert cs.equals(sum_vpolytope_hpolyhedron_, true_sum_vpolytope_hpolyhedron)

        assert cs.equals(sum_hpolyhedron_hpolyhedron, true_sum_hpolyhedron_hpolyhedron)
        assert cs.equals(sum_hpolyhedron_hpolyhedron_, true_sum_hpolyhedron_hpolyhedron)


        # check exceptions: call minkowski_sum instead of __add__
        with self.assertRaises(TypeError):
            I1 + Z1
        with self.assertRaises(TypeError):
            I1 + VP1
        with self.assertRaises(TypeError):
            I1 + HP1
        with self.assertRaises(OtherFunctionError):
            Z1 + I1
        with self.assertRaises(OtherFunctionError):
            Z1 + Z1
        with self.assertRaises(OtherFunctionError):
            Z1 + VP1
        with self.assertRaises(OtherFunctionError):
            Z1 + HP1
        with self.assertRaises(OtherFunctionError):
            VP1 + I1
        with self.assertRaises(OtherFunctionError):
            VP1 + Z1
        with self.assertRaises(OtherFunctionError):
            VP1 + VP1
        with self.assertRaises(OtherFunctionError):
            VP1 + HP1
        with self.assertRaises(OtherFunctionError):
            HP1 + I1
        with self.assertRaises(OtherFunctionError):
            HP1 + Z1
        with self.assertRaises(OtherFunctionError):
            HP1 + VP1
        with self.assertRaises(OtherFunctionError):
            HP1 + HP1

if __name__ == '__main__':
    unittest.main()
    