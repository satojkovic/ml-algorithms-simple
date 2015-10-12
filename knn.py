#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    trainData = np.random.randint(0, 100, (50, 2)).astype(np.float32)
    trainLabel = np.random.randint(0, 2, (50, 1)).astype(np.float32)

    # plot trainData
    red = trainData[trainLabel.ravel() == 0]
    blue = trainData[trainLabel.ravel() == 1]
    plt.scatter(red[:, 0], red[:, 1], c='r')
    plt.scatter(blue[:, 0], blue[:, 1], c='b', marker='x')

    plt.show()


if __name__ == '__main__':
    main()
