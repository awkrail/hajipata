# (8. 1) 線形サポートベクトルマシン
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# アイリスデータセットのロード
iris = load_iris()
X = iris.data
t = iris.target
bias = np.ones((100, 1))
X = X[:100, 2:]
t = t[:100]
epoch = 1000
lr = 1e-15
# ラベルは {-1, 1}にしておく
for i in range(100):
  if t[i] == 0:
    t[i] = -1

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)
# 今回は未定乗数のパラメータaは0とする
N = X_train.shape[0]
a = np.zeros(N)
ones = np.ones(N)

dot_x = np.dot(X_train, X_train.T)
ti_tj = np.dot(t_train.reshape(N, 1), t_train.reshape(1, N))
H = dot_x * ti_tj

def L(a):
  return np.dot(a, ones) - np.dot(a.T, np.dot(H, a))/2.

def dL(a):
  return ones - np.dot(H, a)

# 勾配法でaを最適な求める : 今回は問題がMAXになっているので山登り法
for i in range(epoch):
  a = a + lr * dL(a)
  print("epoch {} loss : {}".format(i, L(a)))

# パラメータwを計算する
w_0, w_1 = 0, 0
W = np.array([w_0, w_1])
for i in range(N):
  W = W + a[i]*t_train[i]*X_train[i]

# テストデータで性能を確認する
test_size = X_test.shape[0]
acc = 0
for idx in range(test_size):
  x = X_test[idx]
  answer = t_test[idx]
  dot_wx = np.dot(W, x)
  if dot_wx >= 0:
    predict = 1
  else:
    predict = -1
  
  if answer == predict:
    acc += 1

print("accuracy : ", acc / test_size)

# テストデータでの結果をプロットしてみる
setosa = X_test[t_test == -1]
versicolour = X_test[t_test == 1]
a = (-1) * (W[0] / W[1])
# b = (-1) * (W[0] / W[2])
b = (1/t_train[0]) - np.dot(W, x)
x = np.arange(0, 6)
y = a * x + b

plt.clf()
plt.scatter(setosa[:, 0], setosa[:, 1], marker='.', color='red')
plt.scatter(versicolour[:, 0], versicolour[:, 1], marker='o', color='blue')
plt.plot(x, y)
plt.title('あやめデータ')
plt.xlabel('花弁の長さ')
plt.ylabel('花弁の幅')
plt.savefig("results/linear_svm.png")
