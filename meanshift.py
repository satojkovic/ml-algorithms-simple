#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import multivariate_normal
from math import sqrt
import sys
from sklearn.cluster import estimate_bandwidth


def euclid_dist(p1, p2):
    if len(p1) != len(p2):
        raise Exception("Mismatch dimension")

    return sqrt(sum([
        (p1[n] - p2[n]) ** 2 for n in range(len(p1))
    ]))


def mean_shift(x, points, bandwidth):
    dists = [euclid_dist(x, p) for p in points]
    distances = np.array(dists).reshape(len(dists), 1)
    weights = np.exp(-1 * (distances ** 2) / (bandwidth ** 2))
    return np.sum(points * weights, axis=0) / np.sum(weights)


def nearest_cluster(mean, cluster_centers):
    nearest = None
    nearest_idx = None
    diff_thresh = 1e-1
    min_dist = sys.float_info.max
    min_center = None
    min_idx = None
    for idx, center in enumerate(cluster_centers):
        dist = euclid_dist(mean, center)
        if dist < min_dist:
            min_dist = dist
            min_center = center
            min_idx = idx

    if min_dist < diff_thresh:
        nearest = min_center
        nearest_idx = min_idx

    return nearest, nearest_idx


def assign_cluster(mean, cluster_centers, points_labels):
    if len(cluster_centers) == 0:
        cluster_centers.append(mean)
        points_labels.append(0)
    else:
        nearest, nearest_idx = nearest_cluster(mean, cluster_centers)
        if nearest is None:
            cluster_centers.append(mean)
            points_labels.append(max(points_labels) + 1)
        else:
            points_labels.append(points_labels[nearest_idx])
    return cluster_centers, points_labels


def mean_shift_clustering(points, bandwidth, max_iterations=300):
    stop_thresh = 1e-3 * bandwidth
    cluster_centers = []
    points_labels = []

    for weighted_mean in points:
        iter = 0
        while True:
            old_mean = weighted_mean
            weighted_mean = mean_shift(old_mean, points, bandwidth)
            converged = euclid_dist(weighted_mean, old_mean) < stop_thresh
            if converged or iter == max_iterations:
                cluster_centers, points_labels = assign_cluster(weighted_mean,
                                                                cluster_centers,
                                                                points_labels)
                break
            iter += 1

    return np.asarray(cluster_centers), np.asarray(points_labels)


def main():
    # sample data
    X = multivariate_normal.load_data()

    # mean shift clustering
    bandwidth = estimate_bandwidth(X, n_samples=500)
    cluster_centers, points_labels = mean_shift_clustering(X, bandwidth)
    print 'Num. of clusters:', len(cluster_centers)
    print cluster_centers
    print points_labels


if __name__ == '__main__':
    main()
