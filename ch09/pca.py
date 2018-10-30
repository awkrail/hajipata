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
if lambdas[0] > lambdas[1]:
  first, second = vectors[:, 0], vectors[:, 1]
else:
  first, second = vectors[:, 1], vectors[:, 0]

# なぜか二つのベクトルが直交しているようには見えない..
plt.quiver(mean[0], mean[1], first[0], first[1], angles='xy',scale_units='xy', scale=1)
plt.quiver(mean[0], mean[1], second[0], second[1], angles='xy',scale_units='xy', scale=1)
plt.scatter(data[:, 0], data[:, 1])
plt.show()