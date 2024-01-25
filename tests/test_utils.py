import unittest
import numpy as np
from continuoussets.utils import comparison, exceptions

class TestUtils(unittest.TestCase):

    def test_compare_matrices(self):
        ''' Test for matrix comparison '''
        # cases:
        # M1 empty x M2
        # M1 x M2 empty
        # M1 x M2 == M1
        # M1 x M2 

        # init matrices
        M1_none = None
        M2_none = None
        M1 = np.array([[2., 1., -1.], [0., 1., -1.]])
        M2 = M1
        M2_zeros = np.array([[2., 1., -1., 0.], [0., 1., -1., 0.]])
        M2_neg = np.array([[-2., 1., 1.], [0., 1., 1.]])
        
        # check results
        assert comparison.compare_matrices(M1_none, M2_none)
        assert not comparison.compare_matrices(M1, M2_none)
        assert comparison.compare_matrices(M1, M2)
        assert not comparison.compare_matrices(M1, M2_zeros, remove_zeros = False)
        assert comparison.compare_matrices(M1, M2_zeros, remove_zeros = True)
        assert not comparison.compare_matrices(M1, M2_neg, check_negation = False)
        assert comparison.compare_matrices(M1, M2_neg, check_negation = True)

    def test_find_aligned_generators(self):
        ''' Test for checking alignment of generators '''
        # cases:
        # - None
        # - no aligned generators
        # - all aligned generators
        # - some aligned generators

        # init matrices
        M_none = None
        M_noaligned = np.array([[0., 1., 2.],[3., 4., 5.]])
        M_allaligned = np.array([[0., 0., 0.],[1., 2., 3.]])
        M_somealigned = np.array([[1., 2., -1., 0., 2., 3., 1.],[-1., 0., 1., 1., 1., 1.5, 0.]])

        # check alignment
        result1 = comparison.find_aligned_generators(M_none)
        result2 = comparison.find_aligned_generators(M_noaligned)
        result3 = comparison.find_aligned_generators(M_allaligned)
        result4 = comparison.find_aligned_generators(M_somealigned)

        # manual computation
        true_result1 = ()
        true_result2 = ()
        true_result3 = ((0, 1, 2), )
        true_result4 = ((0, 2), (1, 6), (4, 5))

        # check results
        assert result1 == true_result1
        assert result2 == true_result2
        assert result3 == true_result3
        assert result4 == true_result4

    def test_OutOfBoundsError(self):
        ''' Test for OutofBoundsError class '''
        n = 2
        valid_range = (1, np.inf)
        given_range = "(0, 10)"
        e = exceptions.OutOfBoundsError(dimension = n,
                                        valid_range = valid_range,
                                        given_range = given_range)
        
        # check that input information is contained in output string
        assert str(n) in e.args[0]
        assert str(valid_range) in e.args[0]
        assert given_range in e.args[0]
    
    def test_EmptySetError(self):
        ''' Test for EmptySetError class '''
        e = exceptions.EmptySetError()

        # check that empty set information is contained in output string
        assert 'empty set' in e.args[0]

    def test_OtherFunctionError(self):
        ''' Test for OtherFunctionError class '''
        args = (1, 2.)
        other_function = 'minkowski_sum'
        e = exceptions.OtherFunctionError(args, other_function)

        # check that error message contains information
        assert 'int' in e.args[0]
        assert 'float' in e.args[0]
        assert other_function in e.args[0]

if __name__ == '__main__':
    unittest.main()
