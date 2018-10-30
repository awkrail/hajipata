# (6.4) ロジスティック回帰
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# アイリスデータセットのロード
iris = load_iris()
X = iris.data
t = iris.target

X = X[:100, 2:]
t = t[:100]

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)

# クラスごとに相関行列を求める
X_0_corrcoef = np.zeros((2, 2))
X_1_corrcoef = np.zeros((2, 2))

for x in X_train[t_train == 0]:
  X_0_corrcoef += x.reshape(-1, 1) @ x.reshape(1, -1)

for x in X_train[t_train == 1]:
  X_1_corrcoef += x.reshape(-1, 1) @ x.reshape(1, -1)

eig_0, eig_vector_0 = np.linalg.eig(X_0_corrcoef)
eig_1, eig_vector_1 = np.linalg.eig(X_1_corrcoef)
# 固有値ベクトルが同じ値..?

eig_0_sorted = np.sort(eig_0)[::-1]
eig_1_sorted = np.sort(eig_1)[::-1]
U_0 = eig_vector_0[:, eig_0.argsort()[::-1]]
U_1 = eig_vector_1[:, eig_1.argsort()[::-1]]

# テストデータで識別する
acc_num = 0
size = X_test.shape[0]


for x, t in zip(X_test, t_test):
  dist_0 = (x @ U_0[:, 0])**2
  dist_1 = (x @ U_1[:, 0])**2

  if dist_0 >= dist_1:
    if t == 0:
      acc_num += 1
  else:
    if t == 1:
      acc_num += 1

print("accuracy : ", acc_num / size)