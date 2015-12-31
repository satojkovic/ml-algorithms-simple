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
    cluster_centers[0] = np.mean(clusters[0], axis=0)

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


def remove_cluster(cidx, clusters, cluster_centers):
    del clusters[cidx]
    del cluster_centers[cidx]


def choose_randomly(cluster):
    """
    choose cluster centers at random.

    Args:
        cluster: sample points
    Returns:
        cluster_centers: shape=(2, 2)
    """
    idx = np.random.randint(0, len(cluster), 2)
    cluster_centers = cluster[idx]
    return cluster_centers


def cosine_similarity(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
    return numerator / denominator if denominator != 0 else 0


def calc_cluster_centers(clusters):
    cluster_centers = defaultdict(np.array)
    for idx in clusters:
        center = np.mean(clusters[idx], axis=0)
        cluster_centers[idx] = center
    return cluster_centers


def bisection(cluster, cluster_center):
    """
    Find two sub-clusters using kmeans algorithm.
    (bisecting step)

    Args:
        cluster:
        cluster_center:
    Returns:
        bisec_clusters:
        bisec_cluster_centers:
    """
    # choose two cluster centers
    bisec_cluster_centers = choose_randomly(cluster)

    # update cluster
    bisec_clusters = defaultdict(list)
    cond = True
    while cond:
        # check if any of the centroids changed
        cond = False

        # find the closest cluster for all samples
        for sample in cluster:
            max_cos_sim = float("-inf")
            max_idx = 0
            for idx, bisec_center in enumerate(bisec_cluster_centers):
                cos_sim = cosine_similarity(sample, bisec_center)
                if max_cos_sim < cos_sim:
                    max_cos_sim = cos_sim
                    max_idx = idx
            bisec_clusters[max_idx].append(sample)

        # check if the centroid changed
        new_centers = calc_cluster_centers(bisec_clusters)
        for bcidx, bisec_center in enumerate(bisec_cluster_centers):
            diff = np.linalg.norm(bisec_center - new_centers[bcidx])
            if  diff > (1 ** -15):
                cond = True

    return bisec_clusters


def append_cluster(max_bisec_cluster, clusters):
    max_id = np.max(clusters.keys())
    for mbcidx in max_bisec_cluster:
        clusters[max_id] = max_bisec_cluster[mbcidx]
        max_id += 1


def overall_similarity(bisec_clusters):
    overall_sim = 0.0
    for cluster in bisec_clusters:
        for cidx, sample in cluster.items():
            overall_sim += np.dot(sample, sample)
    return overall_sim


def repeated_bisection(X, n_clusters, ITER=10):
    # initial cluster contains all samples
    clusters, cluster_centers = init_clusters(X)

    while len(clusters) != n_clusters:
        # choose cluster to split
        cidx, cluster, cluster_center = choose_cluster(clusters,
                                                       cluster_centers)
        # remove chosen cluster from a list
        remove_cluster(cidx, clusters, cluster_centers)

        # do bisecting kmeans
        max_sim = float("-inf")
        max_bisec_clusters = None
        for iter in range(ITER):
            print iter
            # bisecting chosen cluster
            bisec_clusters = bisection(cluster, cluster_center)

            # save bisec_clusters with the highest overall simlarity
            overall_sim = overall_similarity(bisec_clusters)
            if max_sim < overall_sim:
                max_bisec_clusters = bisec_clusters
                max_sim = overall_sim

        # append bisec_cluster to clusters
        append_cluster(max_bisec_clusters, clusters)
        print len(clusters)

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
