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
    return theta, sigma


def log_gaussian_wrap(x, mean, var):
    epsilon = 1.0e-5
    if var < epsilon:
        return 0.0
    return scipy.stats.norm(mean, var).logpdf(x)


def negative_log_likelihood(model, X, y):
    n_features = len(X)
    log_prior_y = -np.log(model['pi'][y])
    log_posterior_x_given_y = -np.sum([log_gaussian_wrap(X[d], model['mean'][y][d], model['var'][y][d]) for d in range(n_features)])
    return log_prior_y + log_posterior_x_given_y


def predict(model, X_test, y_test):
    results = [negative_log_likelihood(model, x, y_test[i])
               for i, x in enumerate(X_test)]
    return results


def main():
    # load sample data
    X, X_labels = multivariate_normal.load_data_with_label()

    # training
    X_train, X_test, y_train, y_test = train_test_split(X, X_labels)
    mean, var = fit(X_train, y_train)
    print 'mean', mean
    print 'var', var

    model = defaultdict(np.array)
    model['mean'] = mean
    model['var'] = var

    # predict
    pred = predict(model, X_test, y_test)

if __name__ == '__main__':
    main()
