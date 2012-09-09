#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import mlpy

np.random.seed(0)
mean, cov, n = [1,5], [[1,1],[1,2]], 300
d = np.random.multivariate_normal(mean, cov, n)
x, y = d[:,0].reshape(-1,1), d[:,1]

ols = mlpy.OLS()
ols.learn(x,y)
xx = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
yy = ols.pred(xx)

fig = plt.figure()
plot = plt.plot(x,y,'o',xx,yy,'--k')
plt.show()
