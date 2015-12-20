#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def choose_randomly(X, n_clusters):
    pass


def section(X, clusters, n_clusters):
    if len(clusters) < n_clusters:
        return

    # choose two cluster centers
    clusters = choose_randomly(X, n_clusters)


def repeated_bisection(X):
    # <cluster_id, cluster_center>
    clusters = defaultdict(np.float)

    # initial two clusters
    section(X, clusters, 2)

    return clusters


def show_clusters(clusters):
    print 'n_clusters = ', len(clusters)
    for i in range(len(clusters)):
        print '** cluster %d:' % (i, clusters[i])


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    clusters = repeated_bisection(X)

    # print results
    show_clusters(clusters)

if __name__ == '__main__':
    main()
