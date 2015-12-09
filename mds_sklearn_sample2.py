#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # load sample data from web
    # https://github.com/cheind/rmds/blob/master/examples/european_city_distances.csv
    df = pd.read_csv('european_city_distances.csv', header=None, sep=';')
    data = df.as_matrix()
    cities = data[:, 0]
    dists = data[:, 1:]
    dists /= np.amax(dists)

    # mds
    mds = MDS(n_components=2, dissimilarity='precomputed')
    results = mds.fit(dists)

if __name__ == '__main__':
    main()
