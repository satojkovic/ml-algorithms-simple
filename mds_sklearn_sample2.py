#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np


def main():
    # load sample data
    data = np.loadtxt('distmat799.txt', delimiter=',')
    dists = data / np.amax(data)

    # mds
    mds = MDS(n_components=2, dissimilarity='precomputed')
    results = mds.fit(dists)

    # plot
    plt.scatter(results.embedding_[:, 0],
                results.embedding_[:, 1],
                marker='o')
    plt.show()

if __name__ == '__main__':
    main()
