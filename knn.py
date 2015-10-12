#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    dtrain = np.random.randint(0, 100, (50, 2)).astype(np.float32)
    dtr_label = np.random.randint(0, 2, (50, 1)).astype(np.float32)
    dtest = np.random.randint(0, 100, (5, 2)).astype(np.float32)

    # plot trainData
    red = dtrain[dtr_label.ravel() == 0]
    blue = dtrain[dtr_label.ravel() == 1]
    plt.scatter(red[:, 0], red[:, 1], c='r')
    plt.scatter(blue[:, 0], blue[:, 1], c='b', marker='x')

    plt.show()


if __name__ == '__main__':
    main()
