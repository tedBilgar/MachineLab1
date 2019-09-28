import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()


data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)


def computeCost(X, y, theta):
    inner = (X * theta - y).T * (X * theta - y)
    return inner[0,0]/(2 * len(X))


theta = np.array([[0],[0]])
print(computeCost(X, y, theta))


#%% алгоритм градиентного спуска
def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = computeCost(X, y, theta)
        theta = theta - alpha / len(X)*X.T*(X*theta-y)

    return theta, cost


g, cost = gradientDescent(X, y, theta, 0.01, 1000)
print(g)



