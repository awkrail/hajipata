# フィッシャーの線形判別式
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

mean1 = np.mean(class_1, axis=0)
mean2 = np.mean(class_2, axis=0)

# フィッシャーの線形判別用のデータ
# Swが分かれば必要なパラメータwが求まる
Sw = np.zeros((2, 2))
for x in class_1:
  Sw += np.dot((x - mean1).reshape(2, 1), (x - mean1).reshape(1, 2))
for x in class_2:
  Sw += np.dot((x - mean2).reshape(2, 1), (x - mean2).reshape(1, 2))
w = np.dot(np.linalg.inv(Sw), (mean1 - mean2).reshape(2, 1))
w1, w2 = w[0][0], w[1][0]

# 切片は明示的にもとまらない : 中点を通ることから求める
m = (mean1 + mean2) / 2
a = -(w1 / w2)
b = m[1] + m[0] * (-a)

# plot
w0, w1, w2 = b, w1, w2
x = np.arange(start=-4, stop=10)
y = a * x + b

plt.scatter(class_1[:, 0], class_1[:, 1], marker='o', color='red')
plt.scatter(class_2[:, 0], class_2[:, 1], marker='o', color='blue')
plt.plot(x, y)
plt.savefig("results/fisher.png")
