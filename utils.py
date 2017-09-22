import numpy as np

def score(array: np.ndarray) -> int:
    '''Quantitatively evaluate the sortedness of a np.ndarray.

    More positive scores are given to arrays in ascending order;
    More negative scores are given to arrays in descending order.

    Args:
        array: Array to be evaluated.

    Returns:
        Score earned by `array`.

    Raises:
        TypeError: If `array` is of a type other than np.ndarray.
        ValueError: If `array` is of a dimension other than 1.

    '''
    if type(array) != np.ndarray:
        raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))
    if array.ndim != 1:
        raise ValueError("argument must be of dimension 1")

    sortedness = 0
    for i in range(array.size):
        [left, n, right] = np.split(array, [i, i+1])
        score += np.sign(n - left).sum()
        score -= np.sign(n - right).sum()
    return sortedness
