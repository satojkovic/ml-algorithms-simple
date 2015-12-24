#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def bisecting(clusters, cluster_centers):
    pass


def choose_randomly(clusters, cluster_id):
    cluster_centers = defaultdict(np.array)
    idx = np.random.randint(0, len(clusters[cluster_id]), 2)
    cluster_centers = clusters[cluster_id][idx]

    # bisecting step
    bisecting(clusters, cluster_centers)

    return cluster_centers


def section(clusters, cluster_id, n_clusters):
    # choose cluster centers
    cluster_centers = choose_randomly(clusters, cluster_id)

    return cluster_centers


def init_clusters(X):
    clusters = defaultdict(np.array)
    cluster_ids = np.zeros(1, dtype=np.int)
    clusters[cluster_ids[0]] = X
    return clusters, cluster_ids


def repeated_bisection(X):
    # clusters<cluster_id, points>
    clusters, cluster_ids = init_clusters(X)

    # split an initial cluster(original data) into two clusters
    # cluster_centers<cluster_id, coordinate_cluster_center>
    cluster_centers = section(clusters, cluster_ids[0], 2)

    return cluster_centers, clusters


def show_clusters(cluster_centers, clusters):
    print '** Result'
    print 'n_clusters = ', len(cluster_centers)
    for i in range(len(cluster_centers)):
        print 'cluster', i, ':', cluster_centers[i]


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    cluster_centers, clusters = repeated_bisection(X)

    # print results
    show_clusters(cluster_centers, clusters)

if __name__ == '__main__':
    main()
