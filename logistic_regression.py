#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split
import numpy as np


def sigmoid(X):
    return (1 / (1 + np.exp(-1 * X)))


def main():
    # load sample data
    X, X_label = multivariate_normal.load_data_with_label()

    # split all data into train and test set
    X_train, X_label_train, X_test, X_label_test = train_test_split(X, X_label)

if __name__ == '__main__':
    main()
