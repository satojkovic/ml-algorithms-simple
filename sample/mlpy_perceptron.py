import numpy as np
import matplotlib.pyplot as plt
import mlpy


## Learning
p = mlpy.Perceptron(alpha=0.1, thr=0.05, maxiters=100) # basic perceptron
p.learn(x, y)

w = p.w()
b = p.bias()
p.err()
p.iters()
xx = np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
yy = - (w[0] * xx + b) / w[1] # separator line
fig = plt.figure(1) # plot
plot1 = plt.plot(x1[:, 0], x1[:, 1], 'ob', x2[:, 0], x2[:, 1], 'or')
plot2 = plt.plot(xx, yy, '--k')
plt.show()
