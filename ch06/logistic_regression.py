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
bias = np.ones((100, 1))
X = X[:100, 2:]
X = np.concatenate((bias, X), axis=1)
t = t[:100]

# 学習データとテストデータに分離
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)
w_0, w_1, w_2 = np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)
W = np.array([w_0, w_1, w_2], dtype=np.float32)

# パラメータの最適化
train_size = X_train.shape[0]
mu = 0.01
epoch = 1000

for e in range(epoch):
  loss = 0
  for idx in range(train_size):
    # 必要なデータの取得
    x = X_train[idx]
    t = t_train[idx]

    # 損失関数と微分の計算
    pi = np.exp(np.dot(W, x.T)) / (1 + np.exp(np.dot(W, x.T)))
    diff_w = (pi - t) * x
    W = W - mu * diff_w

    # Lossを計算する
    loss += (-1) * (t * np.log(pi) + (1 - t) * np.log(1 - pi))
  
  # 学習状況を出力
  print("epoch : ", e, " loss : ", loss)


# テストデータで性能を確認する
test_size = X_test.shape[0]
acc = 0
for idx in range(test_size):
  x = X_test[idx]
  answer = t_test[idx]
  pi = np.exp(np.dot(W, x.T)) / (1 + np.exp(np.dot(W, x.T)))
  if pi >= 0.5:
    predict = 1
  else:
    predict = 0
  
  if answer == predict:
    acc += 1

print("accuracy : ", acc / test_size)

# テストデータでの結果をプロットしてみる
setosa = X_test[t_test == 0]
versicolour = X_test[t_test == 1]

a = (-1) * (W[1] / W[2])
b = (-1) * (W[0] / W[2])
x = np.arange(0, 6)
y = a * x + b

plt.clf()
plt.scatter(setosa[:, 1], setosa[:, 2], marker='.', color='red')
plt.scatter(versicolour[:, 1], versicolour[:, 2], marker='o', color='blue')
plt.plot(x, y)
plt.title('あやめデータ')
plt.xlabel('花弁の長さ')
plt.ylabel('花弁の幅')
plt.savefig("results/logistic_regression.png")