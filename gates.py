import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    '''Performs the sigmoid operation 1/(1 + e^(-x)).

    Args:
        x: Input array.

    Returns:
        Result of the sigmoid function.

    '''
    return 1.0/(1.0 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    '''Performs the ReLU operation f(x)=max(0, x).

    Args:
        x: Input array.

    Returns:
        Result of the ReLU function.

    '''
    orig_shape = x.shape
    x = x.flatten()

    return (np.array([0.0 if n < 0.0 else n for n in x])).reshape(orig_shape)

def dsigmoid(grad: np.ndarray, out: np.ndarray) -> np.ndarray:
    '''Calculates the gradient on a sigmoid unit.

    Args:
        grad: Gradient being backpropogated into the sigmoid unit.
        out: Original output of the sigmoid unit.

    Returns:
        Gradient of the sigmoid unit.

    '''
    return ((1 - out) * out) * grad

def drelu(grad: np.ndarray, out: np.ndarray) -> np.ndarray:
    '''Calculates the gradient on a ReLU unit.

    Args:
        grad: Gradient being backpropogated into the ReLU unit.
        out: Original output of the ReLU unit.

    Returns:
        Gradient of the ReLU unit.

    '''
    prop_grad = grad
    prop_grad[out <= 0] = 0
    return prop_grad
