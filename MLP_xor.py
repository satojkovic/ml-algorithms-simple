#-*- coding: utf-8 -*-

import numpy as np
from MLP import MLP


def main():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    # initialize
    mlp = MLP(inputs, targets, n_hidden_units=3)

    print '--- initialize ---'
    print 'Num of training data: %d' % mlp.ntd
    print 'Num of input layer units: %d' % mlp.nin
    print 'Num of hidden layer units: %d' % mlp.nhid
    print 'Num of output layer units: %d' % mlp.nout
    print 'Shape of first layer weight(v):', mlp.v.shape
    print 'Shape of second layer weight(w):', mlp.w.shape

    # training
    mlp.fit(inputs, targets)
    print '--- training ---'
    print 'first layer weight: '
    print mlp.v
    print 'second layer weight: '
    print mlp.w

    # predict
    print '--- predict ---'
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.predict(i)

if __name__ == '__main__':
    main()
