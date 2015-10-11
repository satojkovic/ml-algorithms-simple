#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def load_data():
    # sample data
    mu1 = [1, 1]
    cov1 = [[4, 0], [30, 100]]
    N1 = 1000
    X1 = np.random.multivariate_normal(mu1, cov1, N1)

    mu2 = [-10, 20]
    cov2 = [[10, 3], [0, 20]]
    N2 = 1000
    X2 = np.random.multivariate_normal(mu2, cov2, N2)

    X = np.r_[X1, X2]

    return X
