#-*- coding: utf-8 -*-

import numpy as np


class Perceptron(object):
    """
    Perceptron Class
    """

    def __init__(self, inputs, targets, T=6, eta=0.1):
        """
        T: number of iterations
        eta: learning rate
        N: number of training data
        m: number of inputs(exclude bias node)
        n: number of neurons
        w: m x n array
        inputs: N x m array
        targets: N x n array
        """
        self._T = T
        self._eta = eta
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.shape[1]
        self._w = np.random.rand(self._m+1, self._n) * 0.1 - 0.05

        bias = - np.ones((self._N, 1))
        self._inputs = np.concatenate((bias, inputs), axis=1)
        self._targets = targets
        self._outputs = np.zeros((self._N, self._n))

        print 'Num of training data: %d' % self._N
        print 'Num of input dim.: %d' % self._m
        print 'Num of output dim.: %d' % self._n

    def train(self):
        """
        a training phase
        """
        for t in xrange(self._T):
            self._outputs = self.recall(self._inputs)
            self._w += self._eta * np.dot(self._inputs.T, self._targets - self._outputs)
        print '--- training phase ---'
        print 'weights:'
        print self._w

    def recall(self, x):
        """
        activataion function

        x: N x m array
        w: m x n array
        """
        y = np.dot(x, self._w)
        return np.where(y > 0, 1, 0)


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [1]])

    p = Perceptron(inputs, targets)
    p.train()

    print '--- recall phase ---'
    inputs_bias = np.concatenate((-np.ones((inputs.shape[0], 1)), inputs), axis=1)
    print p.recall(inputs_bias)

if __name__ == '__main__':
    main()
