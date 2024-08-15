from __future__ import annotations

import inspect
from abc import ABC
from typing import Union

import numpy as np

import continuoussets.utils.tolerances as tol

# implementations of unary/binary operations (only as module!)
# import continuoussets.binary_operations.binary_operations as binary_ops
# import continuoussets.unary_operations.unary_operations as unary_ops
# ! ...yields circular import

# for plotting
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

if __name__ == '__main__':
    print('This is the IConvexSet class.')


# abstract class
class IConvexSet(ABC):

    # flag for validation of input arguments
    validate = True

    # change validation status of input arguments
    @classmethod
    def validate_input_arguments(cls, new_status: bool):
        """Change the flag responsible for whether input arguments are validated or not.

        Args:
            new_status (bool): New validation status.

        Raises:
            TypeError: New validation status must be of type bool.
        """
        if not isinstance(new_status, bool):
            raise TypeError('IConvexSet:validate_input_arguments',
                            'Class attribute validate can only be set to True or False')

        cls.validate = new_status

    # # shortcuts for set equality
    # def __ne__(self, other: Union[IConvexSet, np.ndarray]) -> bool:
    #     """Check whether a set and another set or vector are not equal.

    #     Args:
    #         other (Union[IConvexSet, np.ndarray]): Set of vector.

    #     Returns:
    #         bool: Arguments are not equal.
    #     """
    #     return not binary_ops.equals(self, other)
    
    # def __eq__(self, other: Union[IConvexSet, np.ndarray]) -> bool:
    #     """Check whether a set and another set or vector are not equal.

    #     Args:
    #         other (Union[IConvexSet, np.ndarray]): Set of vector.

    #     Returns:
    #         bool: Arguments are equal.
    #     """
    #     return binary_ops.equals(self, other)
    
    # # unary operations
    # def convert(S: Union[IConvexSet, np.ndarray], set_class: str, *,
    #             mode: str = 'exact') -> IConvexSet:
    #     """Converts a IConvexSet S to another IConvexSet class or a point.

    #     Args:
    #         S (Union[IConvexSet, np.ndarray]): Set or vector.
    #         set_class (str): Name of another IConvexSet class or 'Point'.
    #         mode (str, optional): Approximation of the conversion: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    #     Returns:
    #         bool: Representation possible.
    #     """
    #     return unary_ops.convert(S, set_class, mode = mode)

    # def represents(S: Union[IConvexSet, np.ndarray], set_class: str, *,
    #                rtol: float = tol.REPRESENTS_RTOL, atol: float = tol.REPRESENTS_ATOL) -> bool:
    #     """Checks if a IConvexSet S can also be equivalently represented by another IConvexSet class or a point.

    #     Args:
    #         S (Union[IConvexSet, np.ndarray]): Set or vector.
    #         set_class (str): Name of another IConvexSet class or 'Point'.
    #         rtol (float, optional): Relative tolerance. Defaults to REPRESENTS_RTOL.
    #         atol (float, optional): Absolute tolerance. Defaults to REPRESENTS_ATOL.

    #     Returns:
    #         bool: Representation possible.
    #     """
    #     return unary_ops.represents(S, set_class, rtol = rtol, atol = atol)

    # # predicates
    # def contains(S1: Union[IConvexSet, np.ndarray], S2: Union[IConvexSet, np.ndarray], *,
    #              rtol: float = tol.CONTAINS_RTOL,
    #              atol: float = tol.CONTAINS_ATOL) -> bool:
    #     """Checks containment of an IConvexSet or vector S2 in an IConvexSet S1.
    #     Defined as: forall s2 in S2: s2 in S1?

    #     Args:
    #         other (Union[IConvexSet, np.ndarray]): Set or vector.
    #         rtol (float, optional): Relative tolerance. Defaults to CONTAINS_RTOL.
    #         atol (float, optional): Absolute tolerance. Defaults to CONTAINS_ATOL.

    #     Returns:
    #         bool: Containment.
    #     """
    #     return binary_ops.contains(S1, S2, rtol = rtol, atol = atol)
    
    # def equals(S1: Union[IConvexSet, np.ndarray], S2: Union[IConvexSet, np.ndarray], *,
    #            rtol: float = tol.EQUALS_RTOL,
    #            atol: float = tol.EQUALS_ATOL) -> bool:
    #     """Checks set equality of two IConvexSet derived objects S1 and S2.
    #     Defined as: forall s1 in S1: s1 in S2 and forall s2 in S2: s2 in S1?

    #     Args:
    #         other (Union[IConvexSet, np.ndarray]): Set or vector.
    #         rtol (float, optional): Relative tolerance. Defaults to EQUALS_RTOL.
    #         atol (float, optional): Absolute tolerance. Defaults to EQUALS_ATOL.

    #     Returns:
    #         bool: Set equality.
    #     """
    #     return binary_ops.equals(S1, S2, rtol = rtol, atol = atol)

    # def intersects(S1: Union[IConvexSet, np.ndarray], S2: Union[IConvexSet, np.ndarray], *,
    #                rtol: float = tol.INTERSECTS_RTOL,
    #                atol: float = tol.INTERSECTS_ATOL) -> bool:
    #     """Checks whether the intersection of an IConvexSet or vector S1 with an IConvexSet or vector S2 is non-empty.
    #     Defined as: exists s1 in S1: s1 in S2?

    #     Args:
    #         other (Union[IConvexSet, np.ndarray]): Set or vector.
    #         rtol (float, optional): Relative tolerance. Defaults to INTERSECTS_RTOL.
    #         atol (float, optional): Absolute tolerance. Defaults to INTERSECTS_ATOL.

    #     Returns:
    #         bool: Non-emptiness of intersection.
    #     """
    #     return binary_ops.intersects(S1, S2, rtol = rtol, atol = atol)
    
    # # binary set operations
    # def cartesian_product(S1: Union[IConvexSet, np.ndarray],
    #                       S2: Union[IConvexSet, np.ndarray], *,
    #                       mode: str = 'exact') -> IConvexSet:
    #     """Cartesian product of two IConvexSet or vectors S1 and S2.
    #     Defined as: {[s1^T s2^T]^T | s1 in S1, s2 in S2}.

    #     Args:
    #         S1 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         S2 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         mode (str, optional): Approximation of the result 'inner', 'exact', 'outer'. Defaults to 'exact'.

    #     Returns:
    #         IConvexSet: Cartesian product of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    #     """
    #     return binary_ops.cartesian_product(S1, S2, mode = mode)
    
    # def convex_hull(S1: Union[IConvexSet, np.ndarray],
    #                 S2: Union[IConvexSet, np.ndarray], *,
    #                 mode: str = 'exact') -> IConvexSet:
    #     """Convex hull of two IConvexSet or vectors S1 and S2.
    #     Defined as: {lambda*s1 + (1-lambda)*s2 | s1 in S2, s2 in S2, lambda in [0,1]}.

    #     Args:
    #         S1 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         S2 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         mode (str, optional): Approximation of the result 'inner', 'exact', 'outer'. Defaults to 'exact'.

    #     Returns:
    #         IConvexSet: Convex hull of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    #     """
    #     return binary_ops.convex_hull(S1, S2, mode = mode)
    
    # def minkowski_difference(S1: Union[IConvexSet, np.ndarray],
    #                          S2: Union[IConvexSet, np.ndarray], *,
    #                          mode: str = 'exact') -> IConvexSet:
    #     """Minkowski difference of two IConvexSet or vectors S1 and S2.
    #     Defined as {s | s + S2 in S1}.

    #     Args:
    #         S1 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         S2 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         mode (str, optional): Approximation of the result: 'inner', 'exact', 'outer'. Defaults to 'exact'.

    #     Returns:
    #         IConvexSet: Minkowski difference of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    #     """
    #     return binary_ops.minkowski_difference(S1, S2, mode = mode)
    
    # def minkowski_sum(S1: Union[IConvexSet, np.ndarray],
    #                   S2: Union[IConvexSet, np.ndarray], *,
    #                   mode: str = 'exact') -> IConvexSet:
    #     """Minkowski sum of two IConvexSet or vectors S1 and S2.
    #     Defined as: {s1 + s2 | s1 in S1, s2 in S2}.

    #     Args:
    #         S1 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         S2 (Union[IConvexSet, np.ndarray]): Set or vector.
    #         mode (str, optional): Approximation of the result 'inner', 'exact', 'outer'. Defaults to 'exact'.

    #     Returns:
    #         IConvexSet: Minkowski sum of S1 and S2, of type S1 (unless S1 is np.ndarray, then of type S2).
    #     """
    #     return binary_ops.minkowski_sum(S1, S2, mode = mode)

    # plot
    def plot(self, *, axis: tuple, **kwargs):
        """Plots a 2D projection of a IConvexSet object.

        Args:
            axis (tuple): Subspace on which to project the set for plotting.

        Raises:
            ValueError: Only projections on 2D and 3D supported.
        """
        self._checkSubspace(axis)

        # only support 2D and 3D plot
        if len(axis) not in [2, 3]:
            raise ValueError('IConvexSet:plot',
                             'Only projection onto 2 or 3 axes supported.')

        # project onto axes
        projected_set = self.project(axis = axis)

        # compute vertices
        V = projected_set.vertices()
        if V.shape[0] > 2:
            # correct ordering
            V = V[ConvexHull(V).vertices, :]
            # append first vertex at the end
            V = np.vstack((V, V[0, :]))

        # plot
        if V.shape[0] == 1:
            # single point: add marker
            plt.plot(V[:, 0], V[:, 1], 'o', **kwargs)
        else:
            plt.plot(V[:, 0], V[:, 1], **kwargs)
        plt.show()

    # check functions
    def _checkMode(self, mode: str):
        """Check function for the mode.

        Args:
            mode (str): Mode for a given set operation.

        Raises:
            ValueError: Chosen mode not in ['inner', 'exact', 'outer'].
        """
        if self.validate:
            admissible_modes = ['inner', 'exact', 'outer']
            if mode not in admissible_modes:
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                 f'mode must be in {str(admissible_modes)}')

    def _checkSetClass(self, set_class: str):
        """Check function for choosing another subclass in IConvexSet.

        Args:
            set_class (str): Name of a subclass in IConvexSet or 'Point'.

        Raises:
            ValueError: Chosen class not a subclass of IConvexSet.
        """
        # ensure that 'set_class' argument is the class name of a subclass of IConvexSet
        if self.validate:
            admissible_classes = [cls.__name__ for cls in IConvexSet.__subclasses__()]
            admissible_classes.append('Point')
            if set_class not in admissible_classes:
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                 f'Keyword argument set_class must be in {str(admissible_classes)}')

    def _checkOtherOperand(self, other: Union[IConvexSet, np.ndarray], *, check_dimension: bool = True):
        """Check function for binary set operations.

        Args:
            other (Union[IConvexSet, np.ndarray]): Set or vector.
            check_dimension (bool, optional): Whether unequal dimensions should raise an Exception. Defaults to True.

        Raises:
            TypeError: Other operand must be either a IConvexSet object or an np.ndarray.
            AttributeError: Only vectors (1D np.ndarray) supported.
            AttributeError: Length of np.ndarray does not match dimension of IConvexSet. (Only if check_dimension = True)
            AttributeError: Dimensions of IConvexSet objects do not match. (Only if check_dimension = True)
        """
        if self.validate:
            if not isinstance(other, np.ndarray) and not isinstance(other, IConvexSet):
                # check type
                raise TypeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                f'Other operand must be of type np.ndarray or IConvexSet')
            if isinstance(other, np.ndarray):
                if other.ndim > 1:
                    # assert vector
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                         'If the other operand is an np.ndarray, it must be a 1D np.ndarray')
                elif check_dimension and other.size != self.dimension:
                    # check dimension
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                         f'Dimension of other operand must match dimension of {self.__class__.__name__} object')
            if isinstance(other, IConvexSet) and (check_dimension and self.dimension != other.dimension):
                # check dimension
                raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                     f'Dimension of other operand must match dimension of {self.__class__.__name__} object')

    def _checkSubspace(self, subspace: tuple):
        """Check function for subspaces.

        Args:
            subspace (tuple): Subspace for a projection of a set.

        Raises:
            TypeError: Subspace must be a tuple or list.
            ValueError: The maximum dimension must not exceed the dimension of the set.
            ValueError: The minimum dimension must not be less than 0.
            ValueError: The dimensions must be integer values.
            ValueError: Dimensions must not occur more than once.
        """
        if self.validate:
            if not isinstance(subspace, tuple) and not isinstance(subspace, list):
                raise TypeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                'Keyword argument subspace must be of type tuple or list')
            elif max(subspace) >= self.dimension:
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                 'Keyword argument subspace exceeds dimension of IConvexSet object')
            elif min(subspace) < 0:
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                 'Keyword argument subspace must be positive')
            elif any([np.remainder(d, 1) for d in np.array(subspace)]):
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                 'Keyword argument subspace must be composed of integer values')
            elif np.any(np.array([subspace.count(element) for element in subspace]) > 1):
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                 'Keyword argument subspace must not contain repeated entries')

    def _checkMatrix(self, matrix: np.ndarray):
        """Check function for the left multiplication of matrices on sets.

        Args:
            matrix (np.ndarray): 2D matrix.

        Raises:
            TypeError: Matrix must be a np.ndarray.
            AttributeError: Row dimension of the matrix must match dimension of the set.
        """
        if self.validate:
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                'Matrix must be of type np.ndarray')
            elif isinstance(matrix, np.ndarray):
                if matrix.shape[1] != self.dimension:
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                         'Dimension of matrix does not fit dimension of Interval')
