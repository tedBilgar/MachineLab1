import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Передаем данные для построения графика через либу pandas и их записи в переменную
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#Выводим первые 5 записей и некоторую статистическую информацию о наборе
print(data.head())
data.describe()

#Отображаем (выводим этот график) с определенным размером
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

#Добавляем новую колонку для нашей таблицы слева с именем Ones и со значениями 1
data.insert(0, 'Ones', 1)

#Возвращаем кол-во колонок таблицы
cols = data.shape[1]
#Делим на проекции: X - первые 2 столбца
X = data.iloc[:,0:cols-1]
# y - последний столбец
y = data.iloc[:,cols-1:cols]

#Создаем массив массивов из значений X
X = np.matrix(X.values)
print(X)
#Создаем массив массивов из значений Y
y = np.matrix(y.values)
print(y)


#Реализация формулы для подсчета значения целевой функции
def computeCost(X, y, theta):
   inner = (X * theta - y).T * (X * theta - y)
   print(inner)
   return inner[0,0]/(2 * len(X))


# Обычная проверка на одномерный случай без обобщения
theta = np.array([[0],[0]])
print(computeCost(X, y, theta))


#%% алгоритм градиентного спуска
#Здесь theta определяет входной аргумент для целевой функции, которую мы хотим минимизировать
def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = computeCost(X, y, theta)
        theta = theta - alpha / len(X)*X.T*(X*theta-y)

    return theta, cost


alpha = 0.01
iters = 1000


def createPlot(alpha, iters):
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    print(g)

    x = np.linspace(data.Population.min(),
                    data.Population.max(), 100)
    f = g[0,0] + (g[1, 0] * x)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')

    plt.show()


createPlot(alpha, iters)


#%% Множественная линейная регрессия !!! Могут быть ошибки и неточности далее!!!

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Badroom', 'Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std()
data2.head()

data2.insert(0, 'Ones', 1)

cols = data2.shape[1]
X = data2.iloc[:,0:cols-1]
y = data2.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.array([[0],[0],[0]])
print(computeCost(X, y, theta))

alpha = 0.01
iters = 500

g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()

#Part3

