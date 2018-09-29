# (7.1) パーセプトロン 
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
t = np.array([-1 if t_value == 0 else 1 for t_value in t])


# 学習データとテストデータに分離
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)
w_0, w_1, w_2 = np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)
W = np.array([w_0, w_1, w_2], dtype=np.float32)

miss = True
epoch = 0

while miss:
  miss_count = 0
  miss = False

  for x, t in zip(X_train, t_train):
    f_x = np.dot(W, x)

    if (f_x >= 0 and t >= 0) or (f_x < 0 and t < 0):
      continue

    if (f_x >= 0 and t < 0) or (f_x < 0 and t > 0):
      miss = True
      miss_count += 1
      W = W + t * x
  
  print("epoch ", epoch , " : ", miss_count)


# テストデータで性能を確認する
test_size = X_test.shape[0]
acc = 0
for idx in range(test_size):
  x = X_test[idx]
  answer = t_test[idx]
  f_x = np.dot(W, x)
  if f_x >= 0:
    predict = 1
  else:
    predict = -1
  
  if answer == predict:
    acc += 1

print("accuracy : ", acc / test_size)

# テストデータでの結果をプロットしてみる
setosa = X_test[t_test == -1]
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
plt.savefig("results/perceptron.png")





