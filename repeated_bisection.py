#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def repeated_bisection(X):
    # <cluster_id, cluster_center>
    clusters = defaultdict(np.float)

    return clusters


def print_results(clusters):
    print 'n_clusters = ', len(clusters)
    for i in range(len(clusters)):
        print '** cluster %d:' % (i, clusters[i])


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    clusters = repeated_bisection(X)

    # print results
    print_results(clusters)

if __name__ == '__main__':
    main()
