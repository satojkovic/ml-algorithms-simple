#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal


def repeated_bisection(X):
    # initialize cluster
    centers = []
    X_labels = []

    return centers, X_labels


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # clustering a sample data
    centers, X_labels = repeated_bisection(X)

if __name__ == '__main__':
    main()
