#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def main():
    X, X_labels = multivariate_normal.load_data_with_label()
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    X_labels_uniq = map(np.str, np.unique(X_labels))
    print classification_report(y_test, pred,
                                target_names=X_labels_uniq)


if __name__ == '__main__':
    main()
