# 机器学习练习 7 - K-means 和PCA（主成分分析）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ==================== K-means 聚类 ====================

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx


# 测试函数
data = loadmat('data/ex7data2.mat')
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, initial_centroids)
print('前3个样本最近聚类中心:', idx[0:3])

# 可视化数据
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(data2.head())

sb.set_theme(context="notebook", style="white")
sb.lmplot(x='X1', y='X2', data=data2, fit_reg=False)
plt.show()


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids

print('聚类中心:', compute_centroids(X, idx, 3))


# ==================== 运行 K-means ====================

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids

idx, centroids = run_k_means(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()


# ==================== 初始化聚类中心 ====================

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids

print('随机初始化的聚类中心:', init_centroids(X, 3))


# ==================== K-means 图像压缩 ====================

# 显示原始图像
pic = plt.imread('data/bird_small.png')
plt.imshow(pic)
plt.title('Original Image')
plt.show()

# 使用mat文件中的数据
image_data = loadmat('data/bird_small.mat')
A = image_data['A']
print('图像数据维度:', A.shape)

# normalize value ranges
A = A / 255.

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print('展开后维度:', X.shape)

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int), :]
print('恢复后维度:', X_recovered.shape)

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print('恢复图像维度:', X_recovered.shape)

plt.imshow(X_recovered)
plt.title('Compressed Image (Custom K-means)')
plt.show()


# ==================== 使用 scikit-learn 实现 K-means ====================

# cast to float
pic = plt.imread('data/bird_small.png') / 255.
plt.imshow(pic)
plt.title('Original')
plt.show()

print('图像维度:', pic.shape)

# serialize data
data = pic.reshape(128*128, 3)
print('序列化后维度:', data.shape)

model = KMeans(n_clusters=16, n_init=100)
model.fit(data)

centroids = model.cluster_centers_
print('聚类中心维度:', centroids.shape)

C = model.predict(data)
print('预测结果维度:', C.shape)

compressed_pic = centroids[C].reshape((128, 128, 3))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[0].set_title('Original')
ax[1].imshow(compressed_pic)
ax[1].set_title('Compressed')
plt.show()


# ==================== Principal component analysis（主成分分析） ====================

data = loadmat('data/ex7data1.mat')

X = data['X']

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

U, S, V = pca(X)
print('U:', U, '\nS:', S, '\nV:', V)


# ==================== 投影与恢复 ====================

def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)

Z = project_data(X, U, 1)
print('投影后:', Z)


def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)
print('恢复后:', X_recovered)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()


# ==================== PCA 应用于人脸图像 ====================

faces = loadmat('data/ex7faces.mat')
X = faces['X']
print('人脸数据维度:', X.shape)


def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                 sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

# 渲染一个图像
face = np.reshape(X[3, :], (32, 32))
plt.imshow(face)
plt.title('Original Face')
plt.show()

# 在人脸数据集上运行PCA
U, S, V = pca(X)
Z = project_data(X, U, 100)

# 恢复原来的结构并再次渲染
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3, :], (32, 32))
plt.imshow(face)
plt.title('Recovered Face (100 components)')
plt.show()
