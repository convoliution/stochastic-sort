from typing import Tuple

import numpy as np
import utils
import gates

class Sorter:
    def __init__(self,
                 D: int,
                 H: int = 200,
                 learning_rate: float = 1e-4,
                 decay_rate: float = 0.99,
                 reg_weight: float = 1e-5):
        '''Initializes the properties of the selector networks.

        Args:
            D: Dimensionality of data.
            H: Size of the hidden layer.
            learning_rate: Gradient descent update coefficient.
            decay_rate: Decay coefficient for RMSprop.
            reg_weight: Regularization weight coefficient for loss.

        '''
        self.net1 = Selector(D, H, learning_rate, decay_rate, reg_weight)
        self.net2 = Selector(D, H, learning_rate, decay_rate, reg_weight)

        self.score_history = []
        self.loss_history = [] # stores data loss

    def _data_loss(self, old_score: float, new_score: float) -> float:
        '''Calculates the loss incurred from the prediction.

        Args:
            old_score: Sortedness of the array prior to swap.
            new_score: Sortedness of the array after swap.

        Returns:
            Data loss.

        '''
        return old_score - new_score

    def sort(self, target: np.ndarray) -> np.ndarray:
        self.score_history = []
        self.loss_history = []

        indices = np.arange(target.size)
        mean = np.mean(target)
        dev = np.std(target)
        x = ((target - mean)/dev).reshape((1, target.size))

        self.score_history.append(utils.score(x))

        for _ in range(50000): # hard-coded iterations is temporary for testing
            i1 = self.net1.select(x)
            i2 = self.net2.select(x)

            utils.swap(indices, i1, i2)
            x = x.reshape(-1)[indices].reshape((1, target.size))
            self.score_history.append(utils.score(x))

            data_loss = self._data_loss(self.score_history[-2], self.score_history[-1])
            self.loss_history.append(data_loss)

            self.net1.backprop(data_loss)
            self.net2.backprop(data_loss)

        return target.reshape(-1)[indices].reshape(target.shape)

class Selector:
    def __init__(self,
                 D: int,
                 H: int = 200,
                 learning_rate: float = 1e-4,
                 decay_rate: float = 0.99,
                 reg_weight: float = 1e-5):
        '''Initializes the properties of the selector network.

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

        self.loss_history = [] # stores data loss + regularization loss

        self.weights = { # Glorot & Bengio Xavier initialization
            'W1': np.random.normal(loc=0, scale=(2/(D + H))**0.5, size=(D, H)),
            'W2': np.random.normal(loc=0, scale=(2/(H + D))**0.5, size=(H, D)),
        }
        self.weight_cache = { # for RMSprop
            key: np.zeros_like(value) for key, value in self.weights.items()
        }

        self.cache = () # for backpropogation

        self.pred = None # network's most recent prediction

    def _reg_loss(self) -> float:
        '''Calculates the loss incurred from weight magnitude to discourage overfitting.

        Returns:
            Regularization loss.

        '''
        return 0.5 * self.reg_weight * (np.sum(self.weights['W1']**2) + np.sum(self.weights['W1']**2))

    def _predict(self, prob: np.ndarray) -> int:
        '''Converts the network's output into a prediction.

        Args:
            prob: Array of length D containing values in the range [0.0, 1.0).

        Returns:
            Position of predicted element.

        '''
        if type(prob) != np.ndarray:
            raise TypeError("argument must be of type np.ndarray; argument was type {} instead".format(type(array)))

        self.pred = prob.argmax()
        return self.pred

    def _foreprop(self, x: np.ndarray) -> np.ndarray:
        '''Forward pass through the network.

        Args:
            x: Input data.

        Returns:
            Output of the network.

        '''
        hidd = gates.relu(np.dot(x, self.weights['W1']))
        prob = gates.sigmoid(np.dot(hidd, self.weights['W2']))

        self.cache = (x, hidd, prob)

        return prob

    def backprop(self, data_loss: float):
        '''Backward pass through the network.

        Args:
            data_loss: Loss incurred from the prediction.

        '''
        (x, hidd, prob) = self.cache

        grads = {
            key: np.zeros_like(value) for key, value in self.weights.items()
        }

        loss = data_loss# + self._reg_loss()   What exactly do I do with regularization loss?
        self.loss_history.append(loss)
        dprob = prob.copy()
        pred_mask = np.zeros_like(dprob)
        pred_mask[0][self.pred] = .1
        if loss < 0:
            dprob -= pred_mask
        else:
            dprob -= .1 - pred_mask*2 # encourage all other options
        dsig = gates.dsigmoid(dprob, prob)
        grads['W2'] = np.dot(hidd.T, dsig)
        dhidd = np.dot(dsig, self.weights['W2'].T)
        dre = gates.drelu(dhidd, np.dot(x, self.weights['W1']))
        grads['W1'] = np.dot(x.T, dre)

        for key in self.weights.keys():
            self.weight_cache[key] = (self.decay_rate * self.weight_cache[key]) + ((1 - self.decay_rate) * grads[key]**2)
            self.weights[key] += (-self.learning_rate * grads[key]) / (np.sqrt(self.weight_cache[key]) + 1e-5)

    def select(self, x) -> int:
        prob = self._foreprop(x)
        return self._predict(prob)
