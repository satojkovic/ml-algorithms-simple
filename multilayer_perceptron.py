#-*- coding: utf-8 -*-

import numpy as np


class MLP(object):
    """
    3 Layered Perceptron
    """
    def __init__(self, inputs, targets, hidden_layer_size=2, beta=0.01):
        """
        N: Number of training data
        m: Input layer size
        n: Output layer size
        h: Hidden layer size
        beta: Paramter for sigmoid function
        """
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.shape[1]
        self._h = hidden_layer_size
        self._beta = beta

        self._v = np.random.rand(self._m+1, self._h) * 0.1 - 0.05
        self._w = np.random.rand(self._h+1, self._n) * 0.1 - 0.05

        self._bias = - np.ones((self._N, 1))
        self._inputs = np.concatenate((self._bias, inputs), axis=1)
        self._targets = targets
        self._outputs = np.zeros((self._N, self._n))

        print '--- initialize ---'
        print 'Num of training data: %d' % self._N
        print 'Input layer size: %d' % self._m
        print 'Output layer size: %d' % self._n
        print 'Hidden layer size: %d' % self._h

    def train(self):
        """
        Forward phase:
            Compute the activation of each neuron j in the hidden leyers,
            work through the network until you get to the ouput layers
        Backward phase:
            Compute the error at the output
            Compute the error in the hidden layers
            Update the ouput layer weights
            Update the hidden layer weights
        """
        pass

    def predict(self, x):
        hid = np.dot(x, self._v)
        hid = np.where(self.__actf(hid) > 0, 1, 0)

        hid = np.concatenate((self._bias, hid), axis=1)
        y = np.dot(hid, self._w)
        return np.where(self.__actf(y) > 0, 1, 0)

    def __actf(self, vec):
        """
        Activation function
        """
        return (1 / (1 + np.exp(-self._beta * vec)))


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    mlp = MLP(inputs, targets, hidden_layer_size=2)
    mlp.train()

    print '--- predict phase ---'
    inputs_bias = np.concatenate((-np.ones((inputs.shape[0],
                                            1)), inputs), axis=1)
    print mlp.predict(inputs_bias)

if __name__ == '__main__':
    main()
