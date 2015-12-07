#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # load sample data from web
    # https://github.com/cheind/rmds/blob/master/examples/european_city_distances.csv
    data = pd.read_csv('european_city_distanes.csv',
                       header=0, sep=';')

if __name__ == '__main__':
    main()
