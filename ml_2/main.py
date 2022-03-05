import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#仅适用于简单的逻辑回归，聚类
##主要步骤：插入1列，获得XY，theta，初始值，得出梯度下降函数、逻辑函数以及代价函数，利用梯度下降函数和代价函数进行最优化处理，得到最优的θ，从而得到决策边界
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['grade1', 'grade2', 'admitted'])
print(data.head())
print(data.describe())
(m, n) = data.shape

admitted_1 = data[data.admitted.isin(['1'])]
admitted_2 = data[data.admitted.isin(['0'])]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(admitted_1['grade1'], admitted_1['grade2'], c='b', label="admitted")
ax.scatter(admitted_2['grade1'], admitted_2['grade2'], c='r', marker='x', label='not admitted')
ax.legend(loc=2, bbox_to_anchor=(0.3, 1.12), ncol=2)
ax.set_xlabel('Grade1')
ax.set_ylabel('Grade2')
plt.show()


# 编辑逻辑函数
def logical_f(z):
    return 1 / (1 + np.exp(-z))


def Cost_f(theta, x, y):
    cost_1 = (-y) * np.log(logical_f(x @ theta))
    cost_2 = (1 - y) * np.log(1 - logical_f(x @ theta))
    return np.mean(cost_1 - cost_2)


data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data[['grade1', 'grade2']]
x1 = pd.DataFrame(np.ones((m, 1)), columns=['constant'])
X = pd.concat([x1, X], axis=1)
Y = data['admitted'].astype(float)
X = X.values
Y = Y.values
theta = np.zeros(n)
print(Y.shape)
print(X.shape)
print(theta.shape)
first_cost = Cost_f(theta, X, Y)


def cal_Gradient(theta, X, Y):
    return (X.T @ (logical_f(X @ theta) - Y)) / len(X)


first_Gradient = cal_Gradient(theta, X, Y)

# 根据原函数和求出的梯度函数对θ进行最优化处理
result = opt.fmin_tnc(func=Cost_f, x0=theta, fprime=cal_Gradient, args=(X, Y))
cost_1 = Cost_f(result[0], X, Y)

#预测
def predict(theta, X):
    gx = logical_f(X @ theta)
    return [1 if x>=0.5 else 0 for x in gx]

final_theta=result[0]
pre=predict(final_theta,X)
correct=[1 if a==b else 0 for (a,b) in zip(pre,Y)]
rate=sum(correct)/len(X)
print(rate)

x1 = np.arange(130, step=0.1)
x2 = -(final_theta[0] + x1*final_theta[1]) / final_theta[2]

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(admitted_1['grade1'], admitted_1['grade2'], c='b', label='Admitted')
ax.scatter(admitted_2['grade1'], admitted_2['grade2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x1, x2)
ax.set_xlim(0, 130)
ax.set_ylim(0, 130)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Decision Boundary')
plt.show()