#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split


def train(X_train, y_train):
    pass


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()
    n_labels = max(X_labels)+1
    n_features = len(X[0])

    # training
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)
    train(X_train, y_train)

if __name__ == '__main__':
    main()
