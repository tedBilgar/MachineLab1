import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Передаем данные для построения графика через либу pandas и их записи в переменную
path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Badroom', 'Price'])
#Выводим первые 5 записей и некоторую статистическую информацию о наборе
print(data.head())
data.describe()

data = (data - data.mean()) / data.std()
print(data.head())

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)


def computeCost(X, y, theta):
   inner = (X * theta - y).T * (X * theta - y)
   return inner[0,0]/(2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = computeCost(X, y, theta)
        theta = theta - alpha / len(X)*X.T*(X*theta-y)

    return theta, cost


theta = np.array([[0],[0],[0]])
print('Theta on ZEROs: ' + str(computeCost(X, y, theta)))
alpha = 0.05
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
print('Value of Function on last iterate: ' + str(cost[-1]))


