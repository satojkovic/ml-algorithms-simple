#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split


def main():
    # load sample data
    X, X_label = multivariate_normal.load_data_with_label()
    X_train, X_test, X_label_train, X_label_test = train_test_split(X, X_label)


if __name__ == '__main__':
    main()
