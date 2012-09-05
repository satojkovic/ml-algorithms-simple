#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import mlpy

np.random.seed(0)
mean1, cov1, n1 = [1,5], [[1,1], [1,2]], 200
x1 = np.random.multivariate_normal(mean1, cov1, n1)
y1 = np.ones(n1, dtype=np.int)

mean2, cov2, n2 = [2.5,2.5], [[1,0], [0,1]], 300
x2 = np.random.multivariate_normal(mean2, cov2, n2)
y2 = 2 * np.ones(n2, dtype=np.int)

mean3, cov3, n3 = [5,8], [[0.5,0], [0,0.5]], 200
x3 = np.random.multivariate_normal(mean3, cov3, n3)
y3 = 3 * np.ones(n3, dtype=np.int)

x = np.concatenate((x1,x2,x3), axis=0)
y = np.concatenate((y1,y2,y3))

knn = mlpy.KNN(k=3)
knn.learn(x,y)
xmin, xmax = x[:,0].min()-1, x[:,0].max()+1
ymin, ymax = x[:,1].min()-1, x[:,1].max()+1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))

xnew = np.c_[xx.ravel(), yy.ravel()]
ynew = knn.pred(xnew).reshape(xx.shape)
ynew[ynew==0] = 1

fig = plt.figure()
plot1 = plt.pcolormesh(xx, yy, ynew)
plot2 = plt.scatter(x[:,0], x[:, 1], c=y)
plt.show()
