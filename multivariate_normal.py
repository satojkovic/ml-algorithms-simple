#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


N1 = 1000
N2 = 1000

def load_data():
    # sample data
    mu1 = [1, 1]
    cov1 = [[4, 0], [30, 100]]
    X1 = np.random.multivariate_normal(mu1, cov1, N1)

    mu2 = [-10, 20]
    cov2 = [[10, 3], [0, 20]]
    X2 = np.random.multivariate_normal(mu2, cov2, N2)

    X = np.r_[X1, X2]

    return X


def load_data_with_label():
    # sample data with label
    X = load_data()
    X1_label = np.zeros(N1, dtype=np.int32)
    X2_label = np.ones(N2, dtype=np.int32)
    X_label = np.r_[X1_label, X2_label]

    return X, X_label
