from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

if __name__ == '__main__':
    print('This is the ConvexSet class.')


# abstract class
class ConvexSet(ABC):

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
            raise TypeError('ConvexSet:validate_input_arguments',
                            'Class attribute validate can only be set to True or False')

        cls.validate = new_status

    # negated set equality
    def __ne__(self, other: Union[ConvexSet, np.ndarray]) -> bool:
        """Check whether a set and another set or vector are not equal.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set of vector.

        Returns:
            bool: Arguments are not equal.
        """
        return not self == other

    # plot
    def plot(self, *, axis: tuple, **kwargs):
        """Plots a 2D projection of a ConvexSet object.

        Args:
            axis (tuple): Subspace on which to project the set for plotting.

        Raises:
            ValueError: Only projections on 2D and 3D supported.
        """
        self._checkSubspace(axis)

        # only support 2D and 3D plot
        if len(axis) not in [2, 3]:
            raise ValueError('ConvexSet:plot',
                             'Only projection onto 2 or 3 axes supported.')

        # project onto axes
        projected_set = self.project(axis = axis)

        # compute vertices
        V = projected_set.vertices()
        if V.shape[1] > 2:
            # correct ordering
            V = V[:, ConvexHull(V.T).vertices]
            # append first vertex at the end
            V = np.hstack((V, np.reshape(V[:, 0], (2, 1))))

        # plot
        if V.shape[1] == 1:
            # single point: add marker
            plt.plot(V[0, :], V[1, :], 'o', **kwargs)
        else:
            plt.plot(V[0, :], V[1, :], **kwargs)
        plt.show()

    # conversions
    @abstractmethod
    def interval(self, *, mode: str):
        """Abstract method: Conversion to Interval.

        Args:
            mode (str): Type of conversion: 'inner', 'exact', 'outer'.

        Raises:
            NotImplementedError: Has to be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def zonotope(self, *, mode: str):
        """Abstract method: Conversion to Interval.

        Args:
            mode (str): Type of conversion: 'inner', 'exact', 'outer'.

        Raises:
            NotImplementedError: Has to be implemented in subclasses.
        """
        raise NotImplementedError

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
        """Check function for choosing another subclass in ConvexSet.

        Args:
            set_class (str): Name of a subclass in ConvexSet.

        Raises:
            ValueError: Chosen class not a subclass of ConvexSet.
        """
        # ensure that 'set_class' argument is the class name of a subclass of ConvexSet
        if self.validate:
            admissible_classes = [cls.__name__ for cls in ConvexSet.__subclasses__()]
            if set_class not in admissible_classes:
                raise ValueError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                 f'Keyword argument set_class must be in {str(admissible_classes)}')

    def _checkOtherOperand(self, other: Union[ConvexSet, np.ndarray], *, check_dimension: bool = True):
        """Check function for binary set operations.

        Args:
            other (Union[ConvexSet, np.ndarray]): Set or vector.
            check_dimension (bool, optional): Whether unequal dimensions should raise an Exception. Defaults to True.

        Raises:
            TypeError: Other operand has to be either a ConvexSet object or an np.ndarray.
            AttributeError: Only vectors (1D np.ndarray) supported.
            AttributeError: Length of np.ndarray does not match dimension of ConvexSet. (Only if check_dimension = True)
            AttributeError: Dimensions of ConvexSet objects do not match. (Only if check_dimension = True)
        """
        if self.validate:
            if not isinstance(other, np.ndarray) and not isinstance(other, ConvexSet):
                # check type
                raise TypeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: '
                                f'Other operand must be of type np.ndarray or ConvexSet')
            if isinstance(other, np.ndarray):
                if other.ndim > 1:
                    # assert vector
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                         'If the other operand is an np.ndarray, it must be a 1D np.ndarray')
                elif check_dimension and other.size != self.dimension:
                    # check dimension
                    raise AttributeError(f'{self.__class__.__name__}.{inspect.stack()[1].function}: ',
                                         f'Dimension of other operand must match dimension of {self.__class__.__name__} object')
            if isinstance(other, ConvexSet) and (check_dimension and self.dimension != other.dimension):
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
                                 'Keyword argument subspace exceeds dimension of ConvexSet object')
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
