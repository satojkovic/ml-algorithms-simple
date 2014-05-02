#-*- coding: utf-8 -*-

import numpy as np


class MLP(object):
    """
    Multi Layer Perceptron
    """
    def __init__(self, inputs, targets, num_hidden_neuron):
        """
        N: Number of training data
        m: Number of input neuron
        n: Number of output neuron
        h: Number of hidden neuron
        """
        self._N = inputs.shape[0]
        self._m = inputs.shape[1]
        self._n = targets.shape[1]
        self._h = num_hidden_neuron

        self._w1 = np.random.rand(self._m+1, self._h) * 0.1 - 0.05
        self._w2 = np.random.rand(self._h+1, self._n) * 0.1 - 0.05

        bias1 = - np.ones((self._N, 1))
        self._inputs = np.concatenate((bias1, inputs), axis=1)
        self._targets = targets
        self._outputs = np.zeros((self._N, self._n))

        print 'Num of training data: %d' % self._N
        print 'Num of input dim.: %d' % self._m
        print 'Num of output dim.: %d' % self._n
        print 'Num of hidden nueron: %d' % self._h
        
    def train(self):
        pass

    def recall(self):
        pass


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    
    mlp = MLP(inputs, targets, num_hidden_neuron=2)
    mlp.train()


if __name__ == '__main__':
    main()
