#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def init_clusters(X):
    """
    Init cluster.

    Args:
        X: feature vector
    Returns:
        clusters: cluster<idx, samples>
        cluster_centers: cluster_centers<idx, center>
    """
    clusters = defaultdict(np.array)
    clusters[0] = X

    cluster_centers = defaultdict(np.array)
    idx = np.random.randint(0, len(X), 2)
    cluster_centers[0] = X[idx]

    return clusters, cluster_centers


def choose_cluster(clusters, cluster_centers):
    """
    Choose cluster based on size.

    Args:
        clusters: cluster<idx, samples>
        cluster_centers: cluster_centers<idx, centers>
    Return:
        a cluster including maximum size of samples
    """
    lens = [len(clusters[idx]) for idx in clusters]
    max_idx = np.argsort(np.array(lens))[::-1][0]
    return max_idx, clusters[max_idx], cluster_centers[max_idx]


def repeated_bisection(X, n_clusters):
    # initial cluster contains all samples
    clusters, cluster_centers = init_clusters(X)

    while len(clusters) != n_clusters:
        # choose cluster to split
        cidx, cluster, cluster_center = choose_cluster(clusters, cluster_centers)

        # remove chosen cluster
        del clusters[cidx]
        del cluster_centers[cidx]

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
