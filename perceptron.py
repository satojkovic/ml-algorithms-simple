#-*- coding: utf-8 -*-

import numpy as np


class Perceptron(object):
    """
    Perceptron Class
    """

    def __init__(self, inputs, targets, T=100):
        """
        T: number of iterations
        N: number of training data
        m: number of inputs(exclude bias node)
        n: number of neurons
        w: m x n array
        inputs: N x m array
        targets: N x n array
        """
        self._T = T
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.size[1]
        self._w = np.random.rand(self._m+1, self._n) * 0.1 - 0.5

        bias = - np.ones((self._N, 1))
        self._inputs = np.concatenate((inputs, bias), axis=1)
        self._targets = targets
        self._outputs = np.zeros((self._n, 1))

    def train(self):
        pass

    def recall(self):
        pass


def main():
    p = Perceptron()
    p.train()
    p.recall()

if __name__ == '__main__':
    main()
