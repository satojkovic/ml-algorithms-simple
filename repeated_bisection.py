#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multivariate_normal
from collections import defaultdict
import numpy as np


def cos_similarity(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return numerator / denominator if denominator != 0 else 0.0


def section(clusters, cluster_id, cluster_centers):
    sub_clusters = defaultdict(list)

    # cosine similarity
    for cidx, cluster in clusters.items():
        for sample in cluster:
            max_similarity = -1.0
            max_index = 0
            for ccenter_idx, ccenter in enumerate(cluster_centers):
                cos_sim = cos_similarity(sample, ccenter)
                if cos_sim > max_similarity:
                    max_similarity = cos_sim
                    max_index = ccenter_idx
            sub_clusters[max_index].append(sample)

    return sub_clusters


def choose_randomly(clusters, cluster_id):
    cluster_centers = defaultdict(np.array)
    idx = np.random.randint(0, len(clusters[cluster_id]), 2)
    cluster_centers = clusters[cluster_id][idx]

    return cluster_centers


def bisection(clusters, cluster_id, n_clusters):
    # pick a cluster to split
    cluster_centers = choose_randomly(clusters, cluster_id)

    # find 2-sub clusters using kmeans algorithm(bisecting step)
    sub_clusters = section(clusters, cluster_id, cluster_centers)

    return cluster_centers, sub_clusters


def init_clusters(X):
    clusters = defaultdict(list)
    cluster_ids = np.zeros(1, dtype=np.int)
    clusters[cluster_ids[0]] = X
    return clusters, cluster_ids


def repeated_bisection(X, n_clusters):
    # clusters<cluster_id, points>
    clusters, cluster_ids = init_clusters(X)

    # split an initial cluster(original data) into two clusters
    # cluster_centers<cluster_id, coordinate_cluster_center>
    cluster_centers, clusters = bisection(clusters, cluster_ids[0], 2)

    return cluster_centers, clusters


def show_clusters(cluster_centers, clusters):
    print '** Result'
    print 'n_clusters = ', len(cluster_centers)
    for i in range(len(cluster_centers)):
        print 'cluster_center', i, ':', cluster_centers[i]
        print 'n_samples :', len(clusters[i])


def main():
    # load sample data
    X = multivariate_normal.load_data()

    # Do repeated bisection clustering
    cluster_centers, clusters = repeated_bisection(X, 2)

    # print results
    show_clusters(cluster_centers, clusters)

if __name__ == '__main__':
    main()
