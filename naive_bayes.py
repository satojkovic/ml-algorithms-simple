#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from sklearn.cross_validation import train_test_split
import numpy as np
from collections import defaultdict
import scipy.stats
from sklearn.metrics import classification_report


def fit(X_train, y_train):
    n_labels = max(y_train)+1
    n_samples, n_features = X_train.shape

    classes = np.unique(y_train)
    n_classes = classes.shape[0]

    theta = np.zeros((n_classes, n_features))
    sigma = np.zeros((n_classes, n_features))
    class_prior = np.zeros(n_classes)
    epsilon = 1e-9
    for i, yi in enumerate(classes):
        Xi = X_train[y_train == yi, :]
        theta[i, :] = np.mean(Xi, axis=0)
        sigma[i, :] = np.var(Xi, axis=0) + epsilon
        class_prior[i] = np.float(Xi.shape[0]) / n_samples
    return theta, sigma, class_prior


def predict(model, X_test, y_test):
    classes = np.unique(y_test)
    n_classes = classes.shape[0]
    joint_log_likelihood = []
    for i in range(n_classes):
        jointi = np.log(model['class_prior'][i])
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * model['var'][i, :]))
        n_ij -= 0.5 * np.sum(((X_test - model['mean'][i, :]) ** 2) /
                             (model['var'][i, :]), 1)
        joint_log_likelihood.append(jointi + n_ij)
    joint_log_likelihood = np.array(joint_log_likelihood).T
    return classes[np.argmax(joint_log_likelihood, axis=1)]


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()

    # training
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)
    mean, var, class_prior = fit(X_train, y_train)
    print 'mean', mean
    print 'var', var

    model = defaultdict(np.array)
    model['mean'] = mean
    model['var'] = var
    model['class_prior'] = class_prior

    # predict
    pred = predict(model, X_test, y_test)

    # print result
    X_labels_uniq = map(np.str, np.unique(X_labels))
    print classification_report(y_test, pred,
                                target_names=X_labels_uniq)


if __name__ == '__main__':
    main()
