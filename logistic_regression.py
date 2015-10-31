#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split
import numpy as np


def sigmoid(X):
    return (1 / (1 + np.exp(-1 * X)))


def compute_cost(theta, X, y):
    m = len(X)
    cost = np.sum(
        [yy*np.log(sigmoid(theta.T*xx)) + (1-yy)*np.log(1-sigmoid(theta.T*xx))
         for (xx, yy) in zip(X, y)]
    )
    return (1./m)*cost


def compute_grad(theta, X, y):
    m, dim = X.shape
    grad = np.zeros([dim, 1])
    for j in range(len(theta)):
        grad[j] = np.sum(
            [(sigmoid(theta.T*xx) - yy)*xx[j] for (xx, yy) in zip(X, y)]
        )
        grad[j] = 1./m
    return grad


def main():
    # load sample data
    X, X_label = multivariate_normal.load_data_with_label()

    # split all data into train and test set
    X_train, X_label_train, X_test, X_label_test = train_test_split(X, X_label)

if __name__ == '__main__':
    main()
