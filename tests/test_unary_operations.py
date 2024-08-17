
import unittest
import numpy as np
import continuoussets as cs
from continuoussets import (
    convert, represents,
    equals, contains
)
from continuoussets import (
    Interval, Zonotope, VPolytope, HPolyhedron
)
from continuoussets.utils.exceptions import ExactEvaluationImpossibleError, UnboundedSetError
from continuoussets.utils.comparison import compare_matrices


class TestUnaryOperations(unittest.TestCase):

    def test_Convert(self):
        # cases:
        # - interval: single point
        # - interval: 1D
        # - interval: non-degenerate
        # - interval: degenerate
        # - zonotope: single point
        # - zonotope: 2D box
        # - zonotope: general case
        # - zonotope: degenerate (1D in 2D)
        # - zonotope: degenerate (2D in 3D)
        # - zonotope: 1D
        # - vpolytope: single point
        # - vpolytope: 1D
        # - vpolytope: interval
        # - vpolytope: zonotope
        # - vpolytope: general case
        # - vpolytope: degenerate
        # - hpolyhedron: single point
        # - hpolyhedron: 1D
        # - hpolyhedron: interval
        # - hpolyhedron: zonotope
        # - hpolyhedron: bounded
        # - hpolyhedron: degenerate
        # - hpolyhedron: unbounded
        I_point = Interval(lb = np.array([1.]))
        I_1D = Interval(lb = np.array([1.]), ub = np.array([4.]))
        I_nondeg = Interval(lb = np.array([-1., 0.]), ub = np.array([3., 2.]))
        I_deg = Interval(lb = np.array([-1., 0.]), ub = np.array([-1., 2.]))

        Z_point = Zonotope(c = np.array([1., 0.]))
        Z_1D = Zonotope(c = np.array([2.]),
                        G = np.array([[1.], [-2.], [0.], [-1.]]))
        Z_2D_box = Zonotope(c = np.array([1., 0.]),
                            G = np.array([[1., 0.], [0., -2.]]))
        Z_nondeg = Zonotope(c = np.array([1., -1., 2.]),
                        G = np.array([[1., 1., 0.], [1., 2., -1.], [-2., 0., 1.], [-1., -1., 1.]]))
        Z_2D_deg = Zonotope(c = np.array([1., 0.]),
                            G = np.array([[0., -2.]]))
        Z_3D_deg = Zonotope(c = np.array([2., -1., 1.]),
                            G = np.array([[1., 2., 1.], [-1., 1., 2.], [0., 3., 3.,], [4., 2., -2.], [-1., 4., 5.]]))
        
        VP_point = VPolytope(V = np.array([3., 2., -1.]))
        VP_1D = VPolytope(V = np.array([[-1.], [0.], [3.], [2.]]))
        VP_interval = VPolytope(V = np.array([[-1., 0.], [0., 0.], [0., 2.], [-1., 2.]]))
        VP_zonotope = VPolytope(V = np.array([[1., -6.], [3., -2.], [3., 0.], [1., 2.], [-1., -2.], [-1., -4.]]))
        VP_nondeg = VPolytope(V = np.array([[1., 0.], [0., 1.], [-2., -2.]]))
        VP_deg = VPolytope(V = np.array([[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]))

        HP_point = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.zeros(3))
        HP_1D = HPolyhedron(A = np.array([[1.], [-1.], [1.]]),
                            b = np.array([4., -2., 7.]))
        HP_interval = HPolyhedron(A = np.vstack((np.eye(3), -np.eye(3))), b = np.ones(6))
        HP_zonotope = HPolyhedron(A = np.array([[0., -1.], [-1., -1.], [0.25, -0.5], [0., 1./3.], [1./7., 1./7.], [-0.25, 0.5]]),
                                  b = np.ones(6))
        HP_bounded = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]),
                                 b = np.array([1., 1., 1.]))
        HP_deg = HPolyhedron(A = np.array([[-1., 0., 0.], [0., -1., 0.], [1., 1., 0.], [1., 1., 1.], [-1., -1., -1.]]),
                             b = np.array([0., 0., 1., 1., -1.]))
        HP_unb = HPolyhedron(A = np.array([[1., 0., 0.]]),
                             b = np.array([1.]))
        

        # conversion from interval
        I_point_interval = cs.convert(I_point, 'Interval')
        I_1D_interval = cs.convert(I_1D, 'Interval')
        I_nondeg_interval = cs.convert(I_nondeg, 'Interval')
        I_deg_interval = cs.convert(I_deg, 'Interval')

        I_point_zonotope = cs.convert(I_point, 'Zonotope')
        I_1D_zonotope = cs.convert(I_1D, 'Zonotope')
        I_nondeg_zonotope = cs.convert(I_nondeg, 'Zonotope')
        I_deg_zonotope = cs.convert(I_deg, 'Zonotope')

        I_point_vpolytope = cs.convert(I_point, 'VPolytope')
        I_1D_vpolytope = cs.convert(I_1D, 'VPolytope')
        I_nondeg_vpolytope = cs.convert(I_nondeg, 'VPolytope')
        I_deg_vpolytope = cs.convert(I_deg, 'VPolytope')

        I_point_hpolyhedron = cs.convert(I_point, 'HPolyhedron')
        I_1D_hpolyhedron = cs.convert(I_1D, 'HPolyhedron')
        I_nondeg_hpolyhedron = cs.convert(I_nondeg, 'HPolyhedron')
        I_deg_hpolyhedron = cs.convert(I_deg, 'HPolyhedron')

        # conversions from zonotope
        Z_point_interval = cs.convert(Z_point, 'Interval')
        Z_1D_interval = cs.convert(Z_1D, 'Interval')
        Z_2D_box_interval = cs.convert(Z_2D_box, 'Interval')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(Z_nondeg, 'Interval')
        with self.assertRaises(NotImplementedError):
            cs.convert(Z_nondeg, 'Interval', mode = 'inner')
        Z_nondeg_interval = cs.convert(Z_nondeg, 'Interval', mode = 'outer')
        Z_2D_deg_interval = cs.convert(Z_2D_deg, 'Interval')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(Z_3D_deg, 'Interval')
        Z_3D_deg_interval = cs.convert(Z_3D_deg, 'Interval', mode = 'outer')

        Z_point_zonotope = cs.convert(Z_point, 'Zonotope')
        Z_1D_zonotope = cs.convert(Z_1D, 'Zonotope')
        Z_2D_box_zonotope = cs.convert(Z_2D_box, 'Zonotope')
        Z_nondeg_zonotope = cs.convert(Z_nondeg, 'Zonotope')
        Z_2D_deg_zonotope = cs.convert(Z_2D_deg, 'Zonotope')
        Z_3D_deg_zonotope = cs.convert(Z_3D_deg, 'Zonotope')

        Z_point_vpolytope = cs.convert(Z_point, 'VPolytope')
        Z_1D_vpolytope = cs.convert(Z_1D, 'VPolytope')
        Z_2D_box_vpolytope = cs.convert(Z_2D_box, 'VPolytope')
        Z_nondeg_vpolytope = cs.convert(Z_nondeg, 'VPolytope')
        Z_2D_deg_vpolytope = cs.convert(Z_2D_deg, 'VPolytope')
        Z_3D_deg_vpolytope = cs.convert(Z_3D_deg, 'VPolytope')

        Z_point_hpolyhedron = cs.convert(Z_point, 'HPolyhedron')
        Z_1D_hpolyhedron = cs.convert(Z_1D, 'HPolyhedron')
        Z_2D_box_hpolyhedron = cs.convert(Z_2D_box, 'HPolyhedron')
        Z_nondeg_hpolyhedron = cs.convert(Z_nondeg, 'HPolyhedron')
        Z_2D_deg_hpolyhedron = cs.convert(Z_2D_deg, 'HPolyhedron')
        Z_3D_deg_hpolyhedron = cs.convert(Z_3D_deg, 'HPolyhedron')

        # conversions from vpolytope
        VP_point_interval = cs.convert(VP_point, 'Interval')
        VP_1D_interval = cs.convert(VP_1D, 'Interval')
        VP_interval_interval = cs.convert(VP_interval, 'Interval')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP_zonotope, 'Interval')
        VP_zonotope_interval = cs.convert(VP_zonotope, 'Interval', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP_nondeg, 'Interval')
        VP_nondeg_interval = cs.convert(VP_nondeg, 'Interval', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP_deg, 'Interval')
        VP_deg_interval = cs.convert(VP_deg, 'Interval', mode = 'outer')

        VP_point_zonotope = cs.convert(VP_point, 'Zonotope')
        VP_1D_zonotope = cs.convert(VP_1D, 'Zonotope')
        VP_interval_zonotope = cs.convert(VP_interval, 'Zonotope')
        with self.assertRaises(NotImplementedError):
            cs.convert(VP_zonotope, 'Zonotope')
        VP_zonotope_zonotope = cs.convert(VP_zonotope, 'Zonotope', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP_nondeg, 'Zonotope')
        VP_nondeg_zonotope = cs.convert(VP_nondeg, 'Zonotope', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(VP_deg, 'Zonotope')
        VP_deg_zonotope = cs.convert(VP_deg, 'Zonotope', mode = 'outer')

        VP_point_vpolytope = cs.convert(VP_point, 'VPolytope')
        VP_1D_vpolytope = cs.convert(VP_1D, 'VPolytope')
        VP_interval_vpolytope = cs.convert(VP_interval, 'VPolytope')
        VP_zonotope_vpolytope = cs.convert(VP_zonotope, 'VPolytope')
        VP_nondeg_vpolytope = cs.convert(VP_nondeg, 'VPolytope')
        VP_deg_vpolytope = cs.convert(VP_deg, 'VPolytope')

        VP_point_hpolyhedron = cs.convert(VP_point, 'HPolyhedron')
        VP_1D_hpolyhedron = cs.convert(VP_1D, 'HPolyhedron')
        VP_interval_hpolyhedron = cs.convert(VP_interval, 'HPolyhedron')
        VP_zonotope_hpolyhedron = cs.convert(VP_zonotope, 'HPolyhedron')
        VP_nondeg_hpolyhedron = cs.convert(VP_nondeg, 'HPolyhedron')
        VP_deg_hpolyhedron = cs.convert(VP_deg, 'HPolyhedron')

        # conversions from hpolyhedron
        HP_point_interval = cs.convert(HP_point, 'Interval')
        HP_1D_interval = cs.convert(HP_1D, 'Interval')
        HP_interval_interval = cs.convert(HP_interval, 'Interval')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_zonotope, 'Interval')
        HP_zonotope_interval = cs.convert(HP_zonotope, 'Interval', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_bounded, 'Interval')
        HP_bounded_interval = cs.convert(HP_bounded, 'Interval', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_deg, 'Interval')
        HP_deg_interval = cs.convert(HP_deg, 'Interval', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_unb, 'Interval')

        HP_point_zonotope = cs.convert(HP_point, 'Zonotope')
        HP_1D_zonotope = cs.convert(HP_1D, 'Zonotope')
        HP_interval_zonotope = cs.convert(HP_interval, 'Zonotope')
        with self.assertRaises(NotImplementedError):
            cs.convert(HP_zonotope, 'Zonotope')
        HP_zonotope_zonotope = cs.convert(HP_zonotope, 'Zonotope', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            HP_bounded_zonotope = cs.convert(HP_bounded, 'Zonotope')
        HP_bounded_zonotope = cs.convert(HP_bounded, 'Zonotope', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_deg, 'Zonotope')
        HP_deg_zonotope = cs.convert(HP_deg, 'Zonotope', mode = 'outer')
        with self.assertRaises(ExactEvaluationImpossibleError):
            cs.convert(HP_unb, 'Zonotope')

        # todo: fix
        # HP_point_vpolytope = cs.convert(HP_point, 'VPolytope')
        HP_1D_vpolytope = cs.convert(HP_1D, 'VPolytope')
        HP_interval_vpolytope = cs.convert(HP_interval, 'VPolytope')
        HP_zonotope_vpolytope = cs.convert(HP_zonotope, 'VPolytope')
        HP_bounded_vpolytope = cs.convert(HP_bounded, 'VPolytope')
        HP_deg_vpolytope = cs.convert(HP_deg, 'VPolytope')
        with self.assertRaises(ExactEvaluationImpossibleError):
            HP_unb_vpolytope = cs.convert(HP_unb, 'VPolytope')

        HP_point_hpolyhedron = cs.convert(HP_point, 'HPolyhedron')
        HP_1D_hpolyhedron = cs.convert(HP_1D, 'HPolyhedron')
        HP_interval_hpolyhedron = cs.convert(HP_interval, 'HPolyhedron')
        HP_zonotope_hpolyhedron = cs.convert(HP_zonotope, 'HPolyhedron')
        HP_bounded_hpolyhedron = cs.convert(HP_bounded, 'HPolyhedron')
        HP_deg_hpolyhedron = cs.convert(HP_deg, 'HPolyhedron')
        HP_unb_hpolyhedron = cs.convert(HP_unb, 'HPolyhedron')
        
        # check conversions from interval
        assert cs.equals(I_deg_interval, I_deg)
        assert cs.equals(I_nondeg_interval, I_nondeg)
        assert cs.equals(I_1D_interval, I_1D)
        assert cs.equals(I_point_interval, I_point)

        assert cs.equals(I_deg_zonotope, I_deg)
        assert cs.equals(I_nondeg_zonotope, I_nondeg)
        assert cs.equals(I_1D_zonotope, I_1D)
        assert cs.equals(I_point_zonotope, I_point)

        assert cs.equals(I_deg_vpolytope, I_deg)
        assert cs.equals(I_nondeg_vpolytope, I_nondeg)
        assert cs.equals(I_1D_vpolytope, I_1D)
        assert cs.equals(I_point_vpolytope, I_point)

        assert cs.equals(I_deg_hpolyhedron, I_deg)
        assert cs.equals(I_nondeg_hpolyhedron, I_nondeg)
        assert cs.equals(I_1D_hpolyhedron, I_1D)
        assert cs.equals(I_point_hpolyhedron, I_point)

        # check conversions from zonotope
        assert cs.equals(Z_point_interval, Z_point)
        assert cs.equals(Z_1D_interval, Z_1D)
        assert cs.equals(Z_2D_box_interval, Z_2D_box)
        assert cs.contains(Z_nondeg_interval, Z_nondeg)
        assert cs.equals(Z_2D_deg_interval, Z_2D_deg)
        assert cs.contains(Z_3D_deg_interval, Z_3D_deg)

        assert cs.equals(Z_point_zonotope, Z_point)
        assert cs.equals(Z_1D_zonotope, Z_1D)
        assert cs.equals(Z_2D_box_zonotope, Z_2D_box)
        assert cs.equals(Z_nondeg_zonotope, Z_nondeg)
        assert cs.equals(Z_2D_deg_zonotope, Z_2D_deg)
        assert cs.equals(Z_3D_deg_zonotope, Z_3D_deg)

        assert cs.equals(Z_point_vpolytope, Z_point)
        assert cs.equals(Z_1D_vpolytope, Z_1D)
        assert cs.equals(Z_2D_box_vpolytope, Z_2D_box)
        assert cs.equals(Z_nondeg_vpolytope, Z_nondeg)
        assert cs.equals(Z_2D_deg_vpolytope, Z_2D_deg)
        assert cs.equals(Z_3D_deg_vpolytope, Z_3D_deg)

        assert cs.equals(Z_point_hpolyhedron, Z_point)
        assert cs.equals(Z_1D_hpolyhedron, Z_1D)
        assert cs.equals(Z_2D_box_hpolyhedron, Z_2D_box)
        assert cs.equals(Z_nondeg_hpolyhedron, Z_nondeg)
        assert cs.equals(Z_2D_deg_hpolyhedron, Z_2D_deg)
        assert cs.equals(Z_3D_deg_hpolyhedron, Z_3D_deg)

        # check conversions from vpolytope
        assert cs.equals(VP_point_interval, VP_point)
        assert cs.equals(VP_1D_interval, VP_1D)
        assert cs.equals(VP_interval_interval, VP_interval)
        assert cs.contains(VP_zonotope_interval, VP_zonotope)
        assert cs.contains(VP_nondeg_interval, VP_nondeg)
        assert cs.contains(VP_deg_interval, VP_deg)

        assert cs.equals(VP_point_zonotope, VP_point)
        assert cs.equals(VP_1D_zonotope, VP_1D)
        assert cs.equals(VP_interval_zonotope, VP_interval)
        assert cs.contains(VP_zonotope_zonotope, VP_zonotope)
        assert cs.contains(VP_nondeg_zonotope, VP_nondeg)
        assert cs.contains(VP_deg_zonotope, VP_deg)

        assert cs.equals(VP_point_vpolytope, VP_point)
        assert cs.equals(VP_1D_vpolytope, VP_1D)
        assert cs.equals(VP_interval_vpolytope, VP_interval)
        assert cs.equals(VP_zonotope_vpolytope, VP_zonotope)
        assert cs.equals(VP_nondeg_vpolytope, VP_nondeg)
        assert cs.equals(VP_deg_vpolytope, VP_deg)

        assert cs.equals(VP_point_hpolyhedron, VP_point)
        assert cs.equals(VP_1D_hpolyhedron, VP_1D)
        assert cs.equals(VP_interval_hpolyhedron, VP_interval)
        assert cs.equals(VP_zonotope_hpolyhedron, VP_zonotope)
        assert cs.equals(VP_nondeg_hpolyhedron, VP_nondeg)
        assert cs.equals(VP_deg_hpolyhedron, VP_deg)

        # check conversion from hpolyhedron
        assert cs.equals(HP_point_interval, HP_point)
        assert cs.equals(HP_1D_interval, HP_1D)
        assert cs.equals(HP_interval_interval, HP_interval)
        assert cs.contains(HP_zonotope_interval, HP_zonotope)
        assert cs.contains(HP_bounded_interval, HP_bounded)
        assert cs.contains(HP_deg_interval, HP_deg)

        assert cs.equals(HP_point_zonotope, HP_point)
        assert cs.equals(HP_1D_zonotope, HP_1D)
        assert cs.equals(HP_interval_zonotope, HP_interval)
        assert cs.contains(HP_zonotope_zonotope, HP_zonotope)
        assert cs.contains(HP_bounded_zonotope, HP_bounded)
        assert cs.contains(HP_deg_zonotope, HP_deg)

        # assert cs.equals(HP_point_vpolytope, HP_point)
        assert cs.equals(HP_1D_vpolytope, HP_1D)
        assert cs.equals(HP_interval_vpolytope, HP_interval)
        assert cs.equals(HP_zonotope_vpolytope, HP_zonotope)
        assert cs.equals(HP_bounded_vpolytope, HP_bounded)
        assert cs.equals(HP_deg_vpolytope, HP_deg)

        assert cs.equals(HP_point_hpolyhedron, HP_point)
        assert cs.equals(HP_1D_hpolyhedron, HP_1D)
        assert cs.equals(HP_interval_hpolyhedron, HP_interval)
        assert cs.equals(HP_zonotope_hpolyhedron, HP_zonotope)
        assert cs.equals(HP_bounded_hpolyhedron, HP_bounded)
        assert cs.equals(HP_deg_hpolyhedron, HP_deg)
        assert cs.equals(HP_unb_hpolyhedron, HP_unb)


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
        I_point = Interval(lb = np.array([-2., -1.]))
        I_deg = Interval(lb = np.array([-2., -1.]), ub = np.array([-2., 4.]))
        I_nondeg = Interval(lb = np.array([-2., -1.]), ub = np.array([3., 4.]))
        
        # init zonotopes
        Z_point = Zonotope(c = np.array([1., 0.]))
        Z_1D = Zonotope(c = np.array([1.]), G = np.array([[2.], [-1.]]))
        Z_interval = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0.], [0., -1.], [2., 0.], [0., 0.]]))
        Z_nondeg = Zonotope(c = np.array([1., 0.]), G = np.array([[1., 0.], [-1., 1.]]))

        # init vpolytopes
        VP_point = VPolytope(V = np.array([2., 1.]))
        VP_interval = VPolytope(V = np.array([[-1., 0.], [2., 0.], [2., 1.], [-1., 1.]]))
        VP_zonotope = VPolytope(V = np.array([[4., -1., 1.], [2., -3., 1.], [6., 3., -1.], [4., 1., -1.],
                                       [-2., -3., 3.], [2., 3., 1.], [0., 1., 1.], [2., -3., 3.],
                                       [0., -5., 3.], [4., 1., 1.], [-2., -3., 5.], [-4., -5., 5.],
                                       [0., 1., 3.], [-2., -1., 3.]]))
        VP_nondeg = VPolytope(V = np.array([[2., 1.], [0., 2.], [-1., -2.]]))

        # init hpolyhedra
        HP_point = cs.convert(np.array([2., 1.]), 'HPolyhedron')
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
        assert cs.represents(I_point, 'ndarray')
        assert cs.represents(I_point, 'Interval')
        assert cs.represents(I_point, 'Zonotope')
        assert cs.represents(I_point, 'VPolytope')
        assert cs.represents(I_point, 'HPolyhedron')

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
        assert cs.represents(Z_point, 'ndarray')
        assert cs.represents(Z_point, 'Interval')
        assert cs.represents(Z_point, 'Zonotope')
        assert cs.represents(Z_point, 'VPolytope')
        assert cs.represents(Z_point, 'HPolyhedron')

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
        assert cs.represents(VP_point, 'ndarray')
        assert cs.represents(VP_point, 'Interval')
        assert cs.represents(VP_point, 'Zonotope')
        assert cs.represents(VP_point, 'VPolytope')
        assert cs.represents(VP_point, 'HPolyhedron')

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
        assert cs.represents(HP_point, 'ndarray')
        assert cs.represents(HP_point, 'Interval')
        assert cs.represents(HP_point, 'Zonotope')
        assert cs.represents(HP_point, 'VPolytope')
        assert cs.represents(HP_point, 'HPolyhedron')

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
