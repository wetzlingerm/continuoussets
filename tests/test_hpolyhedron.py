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

        # init HPolyhedron objects
        HP_int = HPolyhedron(A = 1, b = 1)

        # check results
        assert HP_int.dimension == 1

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

if __name__ == '__main__':
    unittest.main()
