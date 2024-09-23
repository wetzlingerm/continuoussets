from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

# import continuoussets.utils.tolerances as tol

# implementations of unary/binary operations (only as module!)
# import continuoussets.binary_operations.binary_operations as binary_ops
# import continuoussets.unary_operations.unary_operations as unary_ops
# ! leads to circular import... fixable?

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

    @abstractmethod
    def copy(self) -> IConvexSet:
        """Returns a deep copy of an IConvexSet object.
        """
        raise NotImplementedError
    
    @abstractmethod
    def __add__(self) -> IConvexSet:
        """Translation of an IConvexSet by a vector.
        """
        raise NotImplementedError

    @abstractmethod
    def __sub__(self) -> IConvexSet:
        """Translation of an IConvexSet by a vector.
        """
        raise NotImplementedError

    @abstractmethod
    def __neg__(self, other) -> IConvexSet:
        """Unary minus operator.
        """
        raise NotImplementedError

    @abstractmethod
    def __pos__(self, other) -> IConvexSet:
        """Unary plus operator.
        """
        raise NotImplementedError

    @abstractmethod
    def basis_affine_hull(self) -> tuple:
        """Computes a basis of the affine hull of an IConvexSet.
        """
        raise NotImplementedError

    @abstractmethod
    def boundary_point(self, direction: np.ndarray, start_point: np.ndarray = None) -> np.ndarray:
        """Computation of the point on the boundary of an IConvexSet in a given direction starting from a given start point.
        """
        raise NotImplementedError

    @abstractmethod
    def bounded(self) -> bool:
        """Checks if an IConvexSet is bounded.
        """
        raise NotImplementedError

    @abstractmethod
    def center(self) -> np.ndarray:
        """Center of an IConvexSet.
        """
        raise NotImplementedError

    @abstractmethod
    def compact(self, *, rtol: float = 1e-12) -> IConvexSet:
        """Minimal representation of an IConvexSet.
        """
        raise NotImplementedError

    @abstractmethod
    def degenerate(self, *, tol: float = 1e-12) -> bool:
        """Determines if an IConvexSet is degenerate.
        """
        raise NotImplementedError
    
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of an IConvexSet object.
        """
        raise NotImplementedError

    @abstractmethod
    def empty(self) -> bool:
        """Checks if an IConvexSet is empty.
        """
        raise NotImplementedError

    @abstractmethod
    def matmul(self, matrix: np.ndarray) -> IConvexSet:
        """Linear map of an IConvexSet S by a matrix M.
        Defined as {M s | s in S}.
        """
        raise NotImplementedError

    @abstractmethod
    def project(self, *, axis: tuple) -> IConvexSet:
        """Projection of an IConvexSet onto a subspace.
        """
        raise NotImplementedError

    @abstractmethod
    def project_affine_hull(self) -> tuple:
        """Projects an IConvexSet onto its own affine hull.
        For degenerate sets, the resulting set is of lower dimension, but non-degenerate.
        """
        raise NotImplementedError

    @abstractmethod
    def support_function(self, direction: np.ndarray) -> tuple[float, np.ndarray]:
        """Support function evaluation of an IConvexSet S in a direction d.
        Value defined as max_{s in S} d^T * s.
        Vector defined as arg max_{s in S} d^T * s.
        """
        raise NotImplementedError

    @abstractmethod
    def vertices(self) -> np.ndarray:
        """Enumeration of all vertices of an IConvexSet.
        """
        raise NotImplementedError

    @abstractmethod
    def volume(self) -> float:
        """Volume computation of an IConvexSet.
        """
        raise NotImplementedError

    # plot
    def plot(self, *, axis: tuple, **kwargs):
        """Plots a 2D projection of an IConvexSet object.

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
            # todo: check degenerate
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

    # # check functions
    # def _checkMode(self, mode: str):
    #     """Check function for the mode.

    #     Args:
    #         mode (str): Mode for a given set operation.

    #     Raises:
    #         ValueError: Chosen mode not in ['inner', 'exact', 'outer'].
    #     """
    #     if self.validate:
    #         admissible_modes = ['inner', 'exact', 'outer']
    #         if mode not in admissible_modes:
    #             raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
    #                              f'mode must be in {str(admissible_modes)}')

    # def _checkSetClass(self, set_class: str):
    #     """Check function for choosing another subclass in IConvexSet.

    #     Args:
    #         set_class (str): Name of a subclass in IConvexSet or 'Point'.

    #     Raises:
    #         ValueError: Chosen class not a subclass of IConvexSet.
    #     """
    #     # ensure that 'set_class' argument is the class name of a subclass of IConvexSet
    #     if self.validate:
    #         admissible_classes = [cls.__name__ for cls in IConvexSet.__subclasses__()]
    #         admissible_classes.append('Point')
    #         if set_class not in admissible_classes:
    #             raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
    #                              f'Keyword argument set_class must be in {str(admissible_classes)}')

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
                elif check_dimension and other.size != self.dimension():
                    # check dimension
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                         f'Dimension of other operand must match dimension of {self.__class__.__name__} object')
            if isinstance(other, IConvexSet) and (check_dimension and self.dimension() != other.dimension()):
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
            elif max(subspace) >= self.dimension():
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
                if matrix.shape[1] != self.dimension():
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                         'Dimension of matrix does not fit dimension of Interval')
