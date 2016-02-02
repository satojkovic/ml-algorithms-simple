#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multivariate_normal
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def dL(L, i, n_samples, X_train, y_train):
    res = 0
    res = np.sum([L[j] * y_train[i] * y_train[j] * np.dot(X_train[i], X_train[j]) for j in range(n_samples)])
    return (1 - res)


def fit(X_train, y_train, learning_rate=0.02, max_iter=50):
    n_samples = X_train.shape[0]
    X_train = np.c_[X_train, np.ones(n_samples)]
    n_features = X_train.shape[1]
    L = np.zeros(n_samples)

    iter = 0
    while iter < max_iter:
        for i in range(n_samples):
            sum =  np.sum([y_train[i] * L[j] * y_train[j]
                           * np.dot(X_train[i], X_train[j])
                           for j in range(n_samples)])
            L[i] = L[i] + learning_rate * (1 - sum)
            L[i] = max(L[i], 0)
        iter += 1

    # calc weight
    w = np.zeros(n_features)
    for i in range(n_samples):
        w += L[i] * y_train[i] * np.array(X_train[i])

    # calc bias term
    S = []
    for i in range(n_samples):
        if L[i] <= 0: continue
        S.append(i)
    sidx = np.random.choice(S)
    b = y_train[sidx] - np.dot(w.T, X_train[sidx])

    plt.plot(X_train[sidx][0], X_train[sidx][1], 'go')

    model = defaultdict(np.array)
    model['L'] = L
    model['w'] = w
    model['b'] = b

    return model


def predict(model, X):
    z = np.zeros(len(X), dtype=np.int)
    pred = model['w'][0] * X[:, 0] + model['w'][1] * X[:, 1] + model['b']
    for i, p in enumerate(pred):
        if p > 0.0:
            z[i] = 0
        else:
            z[i] = 1
    return z


def show_boundary(model, X_train, y_train):
    for i, xt in enumerate(X_train):
        if y_train[i] != 1:
            plt.plot(xt[0], xt[1], 'rx')
        else:
            plt.plot(xt[0], xt[1], 'bx')
    
    x1_min, x1_max = min(X_train[:, 0]) - 1, max(X_train[:, 0]) + 1
    x2_min, x2_max = min(X_train[:, 1]) - 1, max(X_train[:, 1]) + 1
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = [- (model['w'][0] / model['w'][1]) * x - (model['b'] / model['w'][1]) for x in x1]
    plt.plot(x1, x2, 'g-')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    h = 0.1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    plt.show()


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()
    X_labels[X_labels == 0] = -1

    # input data
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)

    # training
    model = fit(X_train, y_train)

    # show boundary
    show_boundary(model, X_train, y_train)


if __name__ == '__main__':
    main()
