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
# ラベルは {-1, 1}にしておく
for i in range(100):
  if t[i] == 0:
    t[i] = -1
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)


# 勾配法でaを最適な求める : 今回は問題がMAXになっているので山登り法
N = X_train.shape[0]
a = np.zeros(N)
ones = np.ones(N)
dot_x = np.dot(X_train, X_train.T)
ti_tj = np.dot(t_train.reshape(N, 1), t_train.reshape(1, N))
H = dot_x * ti_tj

def L(a):
  return np.dot(a, ones) - np.dot(a.T, np.dot(H, a))/2.

beta = 1.0
a_lr = 0.0001
beta_lr = 0.1

for e in range(epoch):
  for i in range(N):
    delta = 1 - (t_train[i] * X_train[i]).dot(a * t_train * X_train.T).sum() - beta * t_train[i] * a.dot(t_train)
    a[i] += a_lr * delta
  for i in range(N):
    # betaを微分して最適化
    beta += beta_lr * a.dot(t_train) ** 2 / 2
  print("epoch {} L(a) : {}".format(e, L(a)))

# パラメータwを計算する
index = a > 0
W = (a * t_train).T.dot(X_train)
b = (t_train[index] - X_train[index].dot(W)).mean()

# テストデータで汎化能力を測定
test_size = X_test.shape[0]
acc = 0
for idx in range(test_size):
  x = X_test[idx]
  answer = t_test[idx]
  dot_wx = np.dot(W, x) + b
  if dot_wx >= 0:
    predict = 1
  else:
    predict = -1
  
  if answer == predict:
    acc += 1

print("accuracy : ", acc / test_size)

setosa = X_train[t_train == -1]
versicolour = X_train[t_train == 1]
x = np.arange(0, 6)
y = -(W[0] * x + b) / W[1]

plt.clf()
plt.scatter(setosa[:, 0], setosa[:, 1], marker='.', color='red')
plt.scatter(versicolour[:, 0], versicolour[:, 1], marker='o', color='blue')
plt.plot(x, y)
plt.title('あやめデータ')
plt.xlabel('花弁の長さ')
plt.ylabel('花弁の幅')
plt.savefig("results/linear_svm_saidainyu.png")
