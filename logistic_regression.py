#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split
import numpy as np
from scipy.optimize import fmin_bfgs


def sigmoid(X):
    OVERFLOW_THRESH = -709
    X = np.sum(X)
    return 0.0 if X < OVERFLOW_THRESH else (1 / (1 + np.exp(-1.0 * X)))


def compute_cost(theta, X, y):
    m = len(X)
    cost = np.sum(
        [yy * np.log(sigmoid(theta.T * xx) + np.finfo(np.float32).eps)
         + (1-yy) * np.log(1-sigmoid(theta.T * xx) + np.finfo(np.float32).eps)
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

    # compute theta
    m, dim = X.shape
    initial_theta = np.zeros([dim, 1], dtype=np.float32)
    theta = fmin_bfgs(compute_cost, initial_theta, args=(X_train,
                                                         X_label_train))
    print theta

if __name__ == '__main__':
    main()
