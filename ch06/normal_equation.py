# 正規方程式
import numpy as np
import matplotlib.pyplot as plt

# 架空のデータセットの用意
# データは正規分布に従って生成
N = 100
mean1 = [1, 3]  # クラス1の平均
mean2 = [3, 1]  # クラス2の平均
cov = [[2.0,0.0], [0.0, 0.1]]  # 共分散行列（2クラス共通)
class_1 = np.random.multivariate_normal(mean1, cov, N//2)
class_2 = np.random.multivariate_normal(mean2, cov, N//2)
bias_x = np.ones((N, 1))
t_1 = np.ones(N//2)
t_2 = np.ones(N//2) * (-1)

# はじめてのパターン認識(P75)通りに変数を指定, 正規方程式からパラメータwを計算
X = np.concatenate((class_1, class_2))
X = np.concatenate((bias_x, X), axis=1)
t = np.concatenate((t_1, t_2)).reshape(N, 1)
w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), t)

# 求めたパラメータから直線を引く
w0, w1, w2 = w[0], w[1], w[2]
x = np.arange(start=-4, stop=10)
y = (((-1) * w1)/w2)*x + ((-1) * w0)/w2

plt.scatter(class_1[:, 0], class_1[:, 1], marker='o', color='red')
plt.scatter(class_2[:, 0], class_2[:, 1], marker='o', color='blue')
plt.plot(x, y)
plt.show()