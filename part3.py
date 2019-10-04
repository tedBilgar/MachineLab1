from sklearn import linear_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)


model = linear_model.LinearRegression()
model.fit(X, y)

f = model.predict(X)
print(f)

x = np.linspace(data.Population.min(),
                data.Population.max(), 100)

f1 = f[0,0] + (f[1, 0] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f1, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()