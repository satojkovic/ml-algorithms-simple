#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def choose_randomly(clusters, n_clusters):
    cluster_centers = defaultdict(np.array)

    for c in range(len(clusters)):
        idx = np.random.randint(0, len(clusters[c]))
        cluster_centers[c] = clusters[c][idx]

    return cluster_centers


def section(clusters, n_clusters):
    # choose two cluster centers
    cluster_centers = choose_randomly(clusters, n_clusters)

    return cluster_centers


def init_clusters(X):
    clusters = defaultdict(list)

    init_cluster_id = 0
    for x in X:
        clusters[init_cluster_id].append(x)

    return clusters


def repeated_bisection(X):
    # <cluster_id, points>
    clusters = init_clusters(X)

    # initial two clusters
    cluster_centers = section(clusters, 2)

    return cluster_centers, clusters


def show_clusters(cluster_centers, clusters):
    print '* n_clusters = ', len(cluster_centers)
    for i in range(len(cluster_centers)):
        print '** cluster', i, ':', cluster_centers[i]


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    cluster_centers, clusters = repeated_bisection(X)

    # print results
    show_clusters(cluster_centers, clusters)

if __name__ == '__main__':
    main()
