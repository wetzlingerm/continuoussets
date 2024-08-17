import unittest
import numpy as np
#import matplotlib.pyplot as plt
import continuoussets as cs
from continuoussets.utils.comparison import compare_matrices
from continuoussets.utils.exceptions import UnboundedSetError, EmptySetError
from continuoussets.convexsets.hpolyhedron import HPolyhedron


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
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        vector = np.array([2., 1.])

        result1 = HP1 + vector

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([4., 2., 0.]))

        assert cs.equals(result1, true_result1)

    def test_neg(self):
        ''' Test for unary minus '''
        # cases:
        # - HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))

        result1 = -HP1

        true_result1 = HPolyhedron(A = np.array([[-1., 0.], [0., -1.], [1., 1.]]), b = np.array([2., 1., 3.]))

        assert cs.equals(result1, true_result1)

    def test_pos(self):
        ''' Test for unary plus '''
        # cases:
        # - HPolyhedron
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        result1 = +HP1

        assert cs.equals(result1, HP1)

    def test_sub(self):
        ''' Test for negative translation '''
        # cases:
        # - HPolyhedron x vector
        HP1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([2., 1., 3.]))
        vector = np.array([2., 1.])

        result1 = HP1 - vector

        true_result1 = HPolyhedron(A = np.array([[1., 0.], [0., 1.], [-1., -1.]]), b = np.array([0., 0., 6.]))

        assert cs.equals(result1, true_result1)

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
        assert compare_matrices(result2, true_result2)
        assert r2 == 1
        assert compare_matrices(result3, true_result2)
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
        with self.assertRaises(UnboundedSetError):
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
        
        with self.assertRaises(EmptySetError):
            HP1.center()
        with self.assertRaises(UnboundedSetError):
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
        assert cs.equals(result1, HP1)
        assert result2.number_constraints() == 3
        assert cs.equals(result2, HP2)
        assert result3.number_constraints() == 1
        assert cs.equals(result3, HP3)
        assert result4.number_constraints() == 3
        assert cs.equals(result4, HP4)

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

        assert cs.equals(result1, HP1)
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result3)
        assert cs.equals(result4, true_result4)

        with self.assertRaises(NotImplementedError):
            HP1.matmul(M5)

    def test_project(self):
        ''' Test for projection '''
        # cases:
        # - 2D -> 1D
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        result1 = HP1.project(axis = (0,))

        true_result1 = HPolyhedron(A = np.array([[1.], [-1.]]), b = np.array([1., 1.]))

        assert cs.equals(result1, true_result1)

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

        assert cs.equals(result1, HP1)
        assert np.array_equal(c1, np.zeros(2))
        assert cs.equals(result2, true_result2)
        assert cs.equals(result3, true_result2)

    def test_reduce(self):
        ''' Test for reduction of the set representation size '''
        # cases:
        # - bounded, non-degenerate
        HP1 = HPolyhedron(A = np.array([[1., 0.], [-1., 1.], [-1., -1.]]), b = np.ones(3))

        with self.assertRaises(NotImplementedError):
            HP1.reduce(order = 2)

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

        assert np.isclose(value1, 3.)
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

        assert compare_matrices(V1, true_result1)
        assert compare_matrices(V2, true_result2)
        
        with self.assertRaises(EmptySetError):
            HP3.vertices()
        with self.assertRaises(UnboundedSetError):
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


if __name__ == '__main__':
    unittest.main()
