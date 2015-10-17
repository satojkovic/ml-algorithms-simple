#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import multivariate_normal
from collections import defaultdict
from math import sqrt


MIN_DIST = 0.0001


def euclid_dist(p_new, p_start):
    return sqrt(sum([
        (p_new[n] - p_start[n]) ** 2 for n in range(len(p_new))
    ]))


def mean_shift(p, points, bandwidth):
    return p


def mean_shift_clustering(points, bandwidth):
    cluster_num = 0
    cluster = defaultdict(list)
    for p in points:
        p_now = p
        while True:
            p_new = mean_shift(p_now, points, bandwidth)
            dist = euclid_dist(p_new, p_now)
            p_now = p_new
            if dist < MIN_DIST:
                break

        # create cluster
        cluster[cluster_num].append(p_now)  # cluster center
        cluster_num += 1

    return cluster


def main():
    # sample data
    X = multivariate_normal.load_data()

    # mean shift clustering
    bandwidth = 3
    cluster = mean_shift_clustering(X, bandwidth)
    print 'Num. of clusters:', len(cluster)


if __name__ == '__main__':
    main()
