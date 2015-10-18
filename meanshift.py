#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import multivariate_normal
from collections import defaultdict
from math import sqrt, pi
import sys


MIN_DIST = 0.0001
MIN_DIST_GROUP = 0.1


def euclid_dist(p1, p2):
    if len(p1) != len(p2):
        raise Exception("Mismatch dimension")

    return sqrt(sum([
        (p1[n] - p2[n]) ** 2 for n in range(len(p1))
    ]))


def dist_to_cluster(pts, cluster):
    min_dist = sys.float_info.max
    min_dist_cluster_idx = None
    for idx, center in cluster.items():
        dist = euclid_dist(pts, center)
        if dist < min_dist:
            dist = min_dist
            min_dist_cluster_idx = idx
    return min_dist_cluster_idx, min_dist


def nearest_cluster(pts, cluster):
    cluster_idx = None
    idx, dist = dist_to_cluster(pts, cluster)
    if dist < MIN_DIST_GROUP:
        cluster_idx = idx
    else:
        cluster_idx = len(cluster) + 1
    return cluster_idx


def gauss_kernel(dist, bandwidth):
    val = (1/(bandwidth*sqrt(2*pi))) * np.exp(-0.5*((dist / bandwidth)) ** 2)
    return val


def mean_shift(p_now, points, bandwidth):
    shift_x, shift_y = 0., 0.
    scale_factor = 0.
    for pts in points:
        dist = euclid_dist(p_now, pts)
        weight = gauss_kernel(dist, bandwidth)
        shift_x += pts[0] * weight
        shift_y += pts[1] * weight
        scale_factor += weight
    shift_x /= scale_factor
    shift_y /= scale_factor
    return np.array([shift_x, shift_y])


def mean_shift_clustering(points, bandwidth):
    cluster = defaultdict(list)
    cluster_pts = defaultdict(list)

    for p in points:
        p_now = p
        while True:
            p_new = mean_shift(p_now, points, bandwidth)
            dist = euclid_dist(p_new, p_now)
            p_now = p_new
            if dist < MIN_DIST:
                break

        # create cluster
        cluster_idx = nearest_cluster(p_new, cluster)
        cluster[cluster_idx] = p_new  # cluster center
        cluster_pts[cluster_idx].append(p_now)

    return cluster, cluster_pts


def main():
    # sample data
    X = multivariate_normal.load_data()

    # mean shift clustering
    bandwidth = 3
    cluster, cluster_pts = mean_shift_clustering(X, bandwidth)
    print 'Num. of clusters:', len(cluster)


if __name__ == '__main__':
    main()
