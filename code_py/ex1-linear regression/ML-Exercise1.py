# 机器学习练习 1 - 线性回归

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==================== 单变量线性回归 ====================

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
print(data.describe())

# 查看数据分布
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()

# 创建以参数θ为特征函数的代价函数
# J(θ) = 1/(2m) * Σ(h_θ(x(i)) - y(i))^2

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# 在训练集中添加一列，以便使用向量化的解决方案计算代价和梯度
data.insert(0, 'Ones', 1)

# 变量初始化
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols-1:cols]  # y是所有行，最后一列

# 检查 X (训练集) 和 y (目标变量)
print(X.head())  # head()是观察前5行
print(y.head())

# 将X和y转换为numpy矩阵，初始化theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

# theta 是一个(1,2)矩阵
print('theta:', theta)

# 看下维度
print('X.shape:', X.shape, 'theta.shape:', theta.shape, 'y.shape:', y.shape)

# 计算代价函数 (theta初始值为0).
print('初始代价:', computeCost(X, y, theta))


# ==================== batch gradient decent（批量梯度下降） ====================
# θ_j := θ_j - α * ∂/∂θ_j * J(θ)

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# 初始化学习速率α和迭代次数
alpha = 0.01
iters = 1000

# 运行梯度下降算法拟合参数θ
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('梯度下降得到的theta:', g)

# 用拟合的参数计算训练模型的代价函数（误差）
print('最终代价:', computeCost(X, y, g))

# 绘制线性模型与数据的拟合图
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 绘制每次迭代的代价变化
# 代价单调递减（凸优化问题）
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# ==================== 多变量线性回归 ====================

# 房屋价格数据集：2个特征（面积、卧室数）和目标（价格）
path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

# 特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

# 重复预处理步骤，对新数据集运行线性回归
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
print('多变量线性回归代价:', computeCost(X2, y2, g2))

# 绘制训练进程
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

# 使用scikit-learn的线性回归函数
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(np.asarray(X), np.asarray(y))

# scikit-learn model的预测表现
x = np.asarray(X)[:, 1]
f = model.predict(np.asarray(X)).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


# ==================== 4. normal equation（正规方程） ====================
# 正规方程解出向量 θ = (X^T * X)^(-1) * X^T * y

def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta

final_theta2 = normalEqn(X, y)  # 感觉和批量梯度下降的theta的值有点差距
print('正规方程得到的theta:', final_theta2)
# 梯度下降得到的结果是matrix([[-3.24140214,  1.1272942 ]])
