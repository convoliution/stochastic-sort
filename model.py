import numpy as np
import tensorflow as tf

class StochasticSorter:

    def _isSorted(self, array: np.ndarray) -> bool:
        '''Evaluates whether or not a np.ndarray is sorted.

        Args:
            array: Array to be evaluated.

        Returns:
            True is `array` is sorted; False if not.

        Raises:
            TypeError: If `array` is of a type other than np.ndarray.

        '''
        if type(array) != np.ndarray:
            raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))

        array = array.flatten()

        for i in range(array.size-1):
            if array[i] > array[i+1]:
                return False
        return True

    def _score(self, array: np.ndarray) -> int:
        '''Quantitatively evaluate the sortedness of a np.ndarray.

        More positive scores are given to arrays in ascending order;
        More negative scores are given to arrays in descending order.

        Args:
            array: Array to be evaluated.

        Returns:
            Score earned by `array`.

        Raises:
            TypeError: If `array` is of a type other than np.ndarray.

        '''
        if type(array) != np.ndarray:
            raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))

        array = array.flatten()

        sortedness = 0
        for i in range(array.size):
            [left, n, right] = np.split(array, [i, i+1])
            sortedness += np.sign(n - left).sum()
            sortedness -= np.sign(n - right).sum()
        return sortedness
