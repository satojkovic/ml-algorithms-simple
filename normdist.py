#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    mu1 = [1, 1]
    cov1 = [[4, 0], [30, 100]]
    N1 = 1000
    X1 = np.random.multivariate_normal(mu1, cov1, N1)

    mu2 = [-10, 20]
    cov2 = [[10, 3], [0, 20]]
    N2 = 1000
    X2 = np.random.multivariate_normal(mu2, cov2, N2)

    plt.scatter(X1[:, 0], X1[:, 1], color='r', marker='x',
                label='$dist_1$')
    plt.scatter(X2[:, 0], X2[:, 1], color='b', marker='x',
                label='$dist_2$')
    plt.show()

if __name__ == '__main__':
    main()
    
