from typing import Tuple

import numpy as np
import utils
import gates

class StochasticSorter:
    def __init__(self,
                 D: int,
                 H: int = 200,
                 learning_rate: float = 1e-4,
                 decay_rate: float = 0.99,
                 reg_weight: float = 1e-5):
        '''Initializes the properties of the sorting network.

        The network possesses the following architecture:
             W1         W2
              \          \
            x-(*)-(ReLU)-(*)-(sigmoid)->prob

        Args:
            D: Dimensionality of data.
            H: Size of the hidden layer.
            learning_rate: Gradient descent update coefficient.
            decay_rate: Decay coefficient for RMSprop.
            reg_weight: Regularization weight coefficient for loss.

        '''
        self.D = D
        self.H = H

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.reg_weight = reg_weight

        self.loss_history = []

        self.weights = { # Glorot & Bengio Xavier initialization
            'W1': np.random.normal(loc=0, scale=(2/(D + H))**0.5, size=(D, H)),
            'W2': np.random.normal(loc=0, scale=(2/(H + 2))**0.5, size=(H, 2)),
        }
        self.weight_cache = { # for RMSprop
            key: np.zeros_like(value) for key, value in self.weights.items()
        }

    def _foreprop(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Forward pass through the network.

        Args:
            x: Input data.

        Returns:
            Tuple containing values required for backpropogation in this format:
                (result of the hidden layer, result of the output layer)

        '''
        hidd = gates.relu(np.dot(x, self.weights['W1']))
        prob = gates.sigmoid(np.dot(hidd, self.weights['W2']))

        return (hidd, prob)

    def _backprop(self, data_loss: float, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        '''Backward pass through the network.

        Args:
            data_loss: Loss incurred from the prediction.
            cache: Tuple containing values required for backpropogation in this format:
                (input data, result of the hidden layer, result of the output layer)

        '''
        (x, hidd, prob) = cache

        grads = {
            key: np.zeros_like(value) for key, value in self.weights.items()
        }

        dprob = prob - data_loss # is this how this works?
        dsig = gates.dsigmoid(dprob, prob)
        grads['W2'] = np.dot(hidd.T, dsig)
        dhidd = np.dot(dsig, self.weights['W2'].T)
        dre = gates.drelu(dhidd, hidd)
        grads['W1'] = np.dot(x.T, dre)

        for key in self.weights.keys():
            self.weight_cache[key] = (self.decay_rate * self.weight_cache[key]) + ((1 - self.decay_rate) * grads[key]**2)
            self.weights[key] += (-self.learning_rate * grads[key]) / (np.sqrt(self.weight_cache[key]) + 1e-5)

    def _loss(self, old_score: float, new_score: float) -> float:
        '''Calculates the sum of the data loss and the regularization loss.

        Args:
            old_score: Sortedness of the array prior to swap.
            new_score: Sortedness of the array after swap.

        Returns:
            Sum of the data loss and the regularization loss.

        '''
        return self._data_loss(old_score, new_score) + self._reg_loss()

    def _data_loss(self, old_score: int, new_score: int) -> int:
        '''Calculates the loss incurred from the prediction.

        Args:
            old_score: Sortedness of the array prior to swap.
            new_score: Sortedness of the array after swap.

        Returns:
            Data loss.

        '''
        return old_score - new_score

    def _reg_loss(self) -> int:
        '''Calculates the loss incurred from weight magnitude to discourage overfitting.

        Returns:
            Regularization loss.

        '''
        return 0.5 * self.reg_weight * (np.sum(self.weights['W1']**2) + np.sum(self.weights['W1']**2))

    def _predict(self, prob: np.ndarray) -> np.ndarray:
        '''Converts the network's output into integer predictions.

        Scales the network's output by the length of the array being sorted,
        then truncates the values into integers.

        Args:
            prob: Array containing values in the range [0.0, 1.0).

        Returns:
            Array with the same length as `prob` containing values in the range [0, D-1]
                where D is the length of the array being sorted.

        '''
        if type(prob) != np.ndarray:
            raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))

        return (prob*self.D).astype(int).flatten()

    def sort(self, x: np.ndarray):
        self.loss_history = []

        x = x.reshape((1, len(x)))
        old_score = utils.score(x)

        print(x)

        old_pred = np.array([-1, -1])
        while True:
            (hidd, prob) = self._foreprop(x)
            pred = self._predict(prob)
            print("Prediction: {}".format(pred))
            if np.array_equal(pred, old_pred):
                break
            utils.swap(x, pred[0], pred[1])
            new_score = utils.score(x)
            loss = self._loss(old_score, new_score)
            self.loss_history.append(loss)
            self._backprop(self._data_loss(old_score, new_score), (x, hidd, prob))

            print(x)
            print("Loss: {}".format(loss))

            old_pred = pred

            old_score = new_score
        print()
