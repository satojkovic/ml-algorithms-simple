#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def main():
    # create sample data
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)


if __name__ == '__main__':
    main()
