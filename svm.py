#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multivariate_normal


def main():
    X, X_labels = multivariate_normal.load_data_with_label()
    n_samples, n_features = X.shape
    X = np.c_[X, np.ones(n_samples)]
    L = np.zeros((n_samples, 1))


if __name__ == '__main__':
    main()
