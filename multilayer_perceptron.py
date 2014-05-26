#-*- coding: utf-8 -*-

import numpy as np


class MLP(object):
    """
    3 Layered Perceptron
    """
    def __init__(self, inputs, targets, hidden_layer_size=2):
        """
        N: Number of training data
        m: Input layer size
        n: Output layer size
        h: Hidden layer size
        """
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.shape[1]
        self._h = hidden_layer_size

        self._v = np.random.rand(self._m+1, self._h) * 0.1 - 0.05
        self._w = np.random.rand(self._h+1, self._n) * 0.1 - 0.05

        self._inputs = inputs
        self._targets = targets
        self._outputs = np.zeros((self._N, self._n))

        print '--- initialize ---'
        print 'Num of training data: %d' % self._N
        print 'Input layer size: %d' % self._m
        print 'Output layer size: %d' % self._n
        print 'Hidden layer size: %d' % self._h

    def fit(self):
        hid = self.__forward(self._v, self._inputs)
        y = self.__forward(self._w, hid)

    def predict(self, x):
        hid = self.__forward(self._v, x)
        y = self.__forward(self._w, hid)
        return y

    def __forward(self, weights, x):
        x = np.concatenate((-np.ones((x.shape[0], 1)), x), axis=1)
        return self.__sigmoid(np.dot(x, weights))

    def __sigmoid(self, z):
        """
        Sigmoid function(Activation function)
        """
        return (1.0 / (1.0 + np.exp(z)))


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    mlp = MLP(inputs, targets, hidden_layer_size=2)
    mlp.fit()

    print '--- predict phase ---'
    print mlp.predict(inputs)

if __name__ == '__main__':
    main()
