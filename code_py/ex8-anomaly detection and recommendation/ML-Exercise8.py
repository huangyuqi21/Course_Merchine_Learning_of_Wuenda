# 机器学习练习 8 - 异常检测和推荐系统

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats
from scipy.optimize import minimize

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ==================== Anomaly detection（异常检测） ====================

data = loadmat('data/ex8data1.mat')
X = data['X']
print('X.shape:', X.shape)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

# 估计高斯分布
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

mu, sigma = estimate_gaussian(X)
print('mu:', mu, 'sigma:', sigma)

# 验证集
Xval = data['Xval']
yval = data['yval']
print('Xval.shape:', Xval.shape, 'yval.shape:', yval.shape)

# 计算概率密度
dist = stats.norm(mu[0], sigma[0])
print('pdf(15):', dist.pdf(15))
print('前50个样本的pdf:', dist.pdf(X[:, 0])[0:50])

# 计算每个值的概率密度
p = np.zeros((X.shape[0], X.shape[1]))
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])
print('p.shape:', p.shape)

# 验证集的概率密度
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])
print('pval.shape:', pval.shape)


# ==================== 选择最佳阈值 ====================

def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
print('epsilon:', epsilon, 'f1:', f1)

# 可视化异常值
outliers = np.where(p < epsilon)
print('outliers:', outliers)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()


# ==================== 协同过滤 ====================

data = loadmat('data/ex8_movies.mat')

Y = data['Y']
R = data['R']
print('Y.shape:', Y.shape, 'R.shape:', R.shape)

# 电影的平均评级
print('电影1的平均评级:', Y[1, np.where(R[1, :] == 1)[0]].mean())

# 可视化评分矩阵
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()


# ==================== 代价函数 ====================

def cost(params, Y, R, num_features):
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initializations
    J = 0

    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    return J


# 测试代价函数
params_data = loadmat('data/ex8_movieParams.mat')
X = params_data['X']
Theta = params_data['Theta']
print('X.shape:', X.shape, 'Theta.shape:', Theta.shape)

users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
Theta_sub = Theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

print('代价:', cost(params, Y_sub, R_sub, features))


# ==================== 带梯度的代价函数 ====================

def cost(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    error = np.multiply((X * Theta.T) - Y, R)
    squared_error = np.power(error, 2)
    J = (1. / 2) * np.sum(squared_error)

    X_grad = error * Theta
    Theta_grad = error.T * X

    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

J, grad = cost(params, Y_sub, R_sub, features)
print('J:', J, 'grad:', grad)


# ==================== 正则化代价函数 ====================

def cost(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    error = np.multiply((X * Theta.T) - Y, R)
    squared_error = np.power(error, 2)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

J, grad = cost(params, Y_sub, R_sub, features, 1.5)
print('正则化 J:', J, 'grad:', grad)


# ==================== 创建自己的电影评分 ====================

movie_idx = {}
f = open('data/movie_ids.txt', encoding='gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

print('电影0:', movie_idx[0])

# 使用练习中提供的评分
ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))


# ==================== 训练协同过滤模型 ====================

R = data['R']
Y = data['Y']

Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)

print('Y.shape:', Y.shape, 'R.shape:', R.shape, 'ratings.shape:', ratings.shape)

movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

print('X.shape:', X.shape, 'Theta.shape:', Theta.shape, 'params.shape:', params.shape)

# 对评分进行归一化
Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))

for i in range(movies):
    idx = np.where(R[i, :] == 1)[0]
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]

print('Ynorm.mean():', Ynorm.mean())

# 训练模型
print('正在训练协同过滤模型...')
fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})
print(fmin)

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

print('X.shape:', X.shape, 'Theta.shape:', Theta.shape)


# ==================== 生成推荐 ====================

predictions = X * Theta.T
my_preds = predictions[:, -1] + Ymean
print('my_preds.shape:', my_preds.shape)

sorted_preds = np.sort(my_preds, axis=0)[::-1]
print('Top 10 ratings:', sorted_preds[:10])

idx = np.argsort(my_preds, axis=0)[::-1]

print("\nTop 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))
