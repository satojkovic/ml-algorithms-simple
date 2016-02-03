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

    # plot decision boundary with meshgrid
    h = 0.1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == '__main__':
    main()
