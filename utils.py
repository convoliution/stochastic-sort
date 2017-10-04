import numpy as np

def isSorted(array: np.ndarray) -> bool:
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

def score(array: np.ndarray) -> float:
    '''Quantitatively evaluate the sortedness of a np.ndarray.

    More positive scores are given to arrays in ascending order;
    more negative scores are given to arrays in descending order.

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

def swap(array: np.ndarray, p1: int, p2: int):
    '''Swaps the elements at two given positions in a np.ndarray after flattening.

    This function flattens the array, swaps the values, then restores the original shape.

    Args:
        array: Array to be manipulated.
        p1: First position.
        p2: Second position.

    Raises:
        TypeError: If `array` is of a type other than np.ndarray.

    '''
    if type(array) != np.ndarray:
        raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))

    orig_shape = array.shape
    array.shape = np.prod(array.shape)

    (array[p2], array[p1]) = (array[p1], array[p2])

    array.shape = orig_shape
