#-*- coding: utf-8 -*-

import numpy as np


class MLP(object):
    """
    3 Layered Perceptron
    """
    def __init__(self, inputs, targets, n_hidden_units=3):
        """
        p: Number of training data
        m: Number of input layer units
        n: Number of output layer units
        h: Number of hidden layer units
        """
        self.ntd = inputs.shape[0]
        self.nin = inputs.shape[1]
        self.nhid = n_hidden_units
        self.nout = targets.size / self.ntd

        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    def fit(self, inputs, targets, learning_rate=0.2, epochs=10000):
        inputs = self.__add_bias(inputs)
        targets = np.array(targets)

        for loop_cnt in xrange(epochs):
            # randomise the order of the inputs
            p = np.random.randint(inputs.shape[0])
            xp = inputs[p]
            bkp = targets[p]

            # forward phase
            gjp = self.__sigmoid(np.dot(self.v, xp))
            gjp = np.insert(gjp, 0, 1)
            gkp = self.__sigmoid(np.dot(self.w, gjp))

            # backward phase(back prop)
            eps2 = self.__sigmoid_deriv(gkp) * (gkp - bkp)
            eps = self.__sigmoid_deriv(gjp) * np.dot(self.w.T, eps2)

            gjp = np.atleast_2d(gjp)
            eps2 = np.atleast_2d(eps2)
            self.w = self.w - learning_rate * np.dot(eps2.T, gjp)

            xp = np.atleast_2d(xp)
            eps = np.atleast_2d(eps)
            self.v = self.v - learning_rate * np.dot(eps.T, xp)[1:, :]

    def predict(self, x):
        x = np.array(x)
        x = np.insert(x, 0, 1)
        hid = self.__sigmoid(np.dot(self.v, x))
        hid = np.insert(hid, 0, 1)
        y = self.__sigmoid(np.dot(self.w, hid))
        return y

    def __add_bias(self, x):
        return np.insert(x, 0, 1, axis=1)

    def __sigmoid(self, u):
        """
        Sigmoid function(Activation function)
        """
        return (1.0 / (1.0 + np.exp(-u)))

    def __sigmoid_deriv(self, u):
        return (u * (1 - u))


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
