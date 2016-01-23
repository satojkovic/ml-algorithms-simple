#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multivariate_normal
from sklearn.cross_validation import train_test_split


def fit(X_train, y_train):
    pass


def predict(model, X_test, y_test):
    pass


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()

    # input data
    n_samples, n_features = X.shape
    X = np.c_[X, np.ones(n_samples)]
    L = np.zeros((n_samples, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)

    # training
    model = fit(X_train, y_train)


if __name__ == '__main__':
    main()
