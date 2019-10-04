from sklearn import linear_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


model = linear_model.LinearRegression()
model.fit(X, y)

f = model.predict(X)
print(f[-1])
