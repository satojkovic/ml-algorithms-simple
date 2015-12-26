#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def cos_similarity(v1, v2):
    numerator = np.sum([e1 * e2 for e1 in v1 for e2 in v2])
    denominator = np.sqrt(np.sum([v * v for v in v1]) *
                          np.sum([v * v for v in v2]))
    return numerator / denominator if denominator != 0 else 0.0


def bisecting(clusters, cluster_id, cluster_centers):
    sub_clusters = defaultdict(list)

    # cosine similarity
    for cidx, cluster in clusters.items():
        max_similarity = -1.0
        max_index = 0
        for ccenter_idx, ccenter in enumerate(cluster_centers):
            cos_sim = cos_similarity(cluster, ccenter)
            if cos_sim > max_similarity:
                max_similarity = cos_sim
                max_index = ccenter_idx
        sub_clusters[max_index].append(cluster)

    return sub_clusters


def choose_randomly(clusters, cluster_id):
    cluster_centers = defaultdict(np.array)
    idx = np.random.randint(0, len(clusters[cluster_id]), 2)
    cluster_centers = clusters[cluster_id][idx]

    return cluster_centers


def section(clusters, cluster_id, n_clusters):
    # pick a cluster to split
    cluster_centers = choose_randomly(clusters, cluster_id)

    # find 2-sub clusters using kmeans algorithm(bisecting step)
    sub_clusters = bisecting(clusters, cluster_id, cluster_centers)

    return cluster_centers, sub_clusters


def init_clusters(X):
    clusters = defaultdict(list)
    cluster_ids = np.zeros(1, dtype=np.int)
    clusters[cluster_ids[0]] = X
    return clusters, cluster_ids


def repeated_bisection(X):
    # clusters<cluster_id, points>
    clusters, cluster_ids = init_clusters(X)

    # split an initial cluster(original data) into two clusters
    # cluster_centers<cluster_id, coordinate_cluster_center>
    cluster_centers, clusters = section(clusters, cluster_ids[0], 2)

    return cluster_centers, clusters


def show_clusters(cluster_centers, clusters):
    print '** Result'
    print 'n_clusters = ', len(cluster_centers)
    for i in range(len(cluster_centers)):
        print 'cluster_center', i, ':', cluster_centers[i]
        print clusters[i]


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    cluster_centers, clusters = repeated_bisection(X)

    # print results
    show_clusters(cluster_centers, clusters)

if __name__ == '__main__':
    main()
