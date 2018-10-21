import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

args = sys.argv
# 架空のデータセットの用意
# データは正規分布に従って生成
if len(args) == 2 and args[1] == "generate":
  N = 100
  mean1 = [1, 3]  # クラス1の平均
  cov = [[1.0,0.5], [0.5, 3.0]]  # 共分散行列（2クラス共通)
  class_1 = np.random.multivariate_normal(mean1, cov, N//2)
  plt.scatter(class_1[:, 0], class_1[:, 1])
  plt.show()
  plt.savefig("pca.png")
  with open("pca.pkl", "wb") as f:
    pickle.dump(class_1, f)

with open("pca.pkl", "rb") as f:
  data = pickle.load(f)

# 平均を軸ごとに求める
mean = data.mean(axis=0)
X = data - mean
N = data.shape[0]

var = (1/N) * np.dot(X.T, X)
lambdas, vectors = np.linalg.eig(var)
# TODO : 寄与順にソートする, ベクトルを描写する
first = vectors[:, 0]
second = vectors[:, 1]

print(np.dot(first, second))

"""
x = np.arange(-1, 4)
y = (first[1] / first[0])*x
plt.plot(x, y, color="green")

x = np.arange(-1, 5)
y = (second[1] / second[0])*x
plt.plot(x, y, color="red")

plt.scatter(data[:, 0], data[:, 1])
plt.show()
"""