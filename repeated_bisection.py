#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def repeated_bisection(X, n_clusters):
    # initial cluster contains all samples
    clusters = defaultdict(np.ndarray)
    clusters[0] = X

    return clusters


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # do repeated bisection clustering
    n_clusters = 2
    clusters = repeated_bisection(X, n_clusters)

    # show results
    print '** Results'
    print 'n_clusters:', len(clusters)

if __name__ == '__main__':
    main()
