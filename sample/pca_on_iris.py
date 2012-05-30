#!/usr/bin/env python

import numpy as np
import mlpy
import matplotlib.pyplot as plt

iris = np.loadtxt('iris.csv', delimiter=',')
x, y = iris[:,:4], iris[:,4].astype(np.int) # x: observation attributes, y: classes

pca = mlpy.PCA()
pca.learn(x)
z = pca.transform(x, k=2) # k=2 dimentional subspace

fig1 = plt.figure(1)
title = plt.title("PCA on iris dataset")
plot = plt.scatter(z[:,0], z[:,1], c=y)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")

plt.show()
