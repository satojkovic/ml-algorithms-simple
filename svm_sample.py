#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


def main():
    X, X_labels = multivariate_normal.load_data_with_label()
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)

    clf = SVC()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    X_labels_uniq = map(np.str, np.unique(X_labels))
    print classification_report(y_test, pred,
                                target_names=X_labels_uniq)

    # plot test data
    colors = ['r', 'b']
    for i in range(2):
        x, y = X_test[y_test == i, 0], X_test[y_test == i, 1]
        plt.scatter(x, y, color=colors[i], marker='x')
    plt.show()

if __name__ == '__main__':
    main()
