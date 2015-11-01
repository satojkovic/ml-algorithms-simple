#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(theta, X, y):
    m = len(X)
    J = np.sum(
        [(np.dot(theta.T, xx) - yy) ** 2 for (xx, yy) in zip(X, y)]
    )
    return (1./2*m) * J


def gradient_descent(theta, X, y, alpha, max_iter):
    m = len(X)
    num_params = len(theta)
    grad = np.zeros([num_params, 1])

    iter = 0
    while iter < max_iter:
        for j in range(num_params):
            grad[j] = np.sum(
                [(np.dot(theta.T, xx) - yy) * xx[j] for (xx, yy) in zip(X, y)]
            )
            grad[j] = (alpha/m) * grad[j]
            theta[j] -= grad[j]
        iter += 1

    return theta


def main():
    # load sample data
    data = multivariate_normal.load_data_single()
    X_, y = data[:, 0], data[:, 1]
    X = np.ones([y.size, 2])
    X[:, 1] = X_

    # compute theta
    m, dim = X.shape
    theta = np.zeros([dim, 1])
    alpha, max_iter = 0.01, 300
    theta = gradient_descent(theta, X, y, alpha, max_iter)
    print theta

    # plot sample data and predicted line
    plt.scatter(data[:, 0], data[:, 1], color='r', marker='x')
    xx = np.linspace(-10, 10)
    yy = theta[0] + theta[1] * xx
    plt.plot(xx, yy, 'k-')
    plt.show()


if __name__ == '__main__':
    main()
