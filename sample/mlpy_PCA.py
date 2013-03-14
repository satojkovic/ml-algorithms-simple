#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import mlpy

def main():
    np.random.seed(0)
    mean, cov, n = [0,0], [[1,1], [1,1.5]], 100
    x = np.random.multivariate_normal(mean, cov, n)

    pca = mlpy.PCA('svd', 0)
    pca.learn(x)
    coeff = pca.coeff()

    fig = plt.figure(1)
    plot1 = plt.plot(x[:,0], x[:,1], 'o')
    plot2 = plt.plot([0, coeff[0,0]], [0, coeff[0,1]], linewidth=4, color='r')
    plot2 = plt.plot([0, coeff[1,0]], [0, coeff[1,1]], linewidth=4, color='g')
    xx = plt.xlim(-4,4)
    yy = plt.ylim(-4,4)
    plt.show()

if __name__ == '__main__':
    main()
