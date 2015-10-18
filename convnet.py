#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference: http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from urllib import urlretrieve
import cPickle as pickle
import os
import gzip

import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = os.path.basename(url)
    if not os.path.exists(filename):
        print 'Downloading MNIST dataset...'
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)

    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    y_train = y_train.astype(np.int)
    y_val = y_val.astype(np.int)
    y_test = y_test.astype(np.int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    plt.imshow(X_train[0][0], cmap=cm.binary)


if __name__ == '__main__':
    main()
