import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# アイリスデータセットのロード
iris = load_iris()
X = iris.data
t = iris.target

X = X[:150, 2:]
t = t[:150]

#X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)

# k-means
k_num = 3

"""
結構初期値に依存するかも?
"""
mu_s = [np.random.rand(2) for _ in range(k_num)]
delta = 10000000
eps = 0.5

while delta > eps:
  """
  まず, どこの点に属するか決める
  """
  all_array = [
    [],
    [],
    []
  ]
  for x in X:
    dist_0 = np.sum(np.square(x  - mu_s[0]))
    dist_1 = np.sum(np.square(x - mu_s[1]))
    dist_2 = np.sum(np.square(x - mu_s[2]))
    min_index = np.argmin(np.array([dist_0, dist_1, dist_2]))
    all_array[min_index].append(x.tolist())
  
  all_array = np.array(all_array)
  """
  muを更新する
  この時に更新量をdeltaに入れておく

  TODO : 家に帰ったら書く
  """
  new_mus = []
  for i, array in enumerate(all_array):
    if len(array) == 0:
      new_mus.append(mu_s[i])
    else:
      new_mus.append(np.mean(array, axis=0))
  
  mu_s = np.array(mu_s)
  new_mus = np.array(new_mus)
  delta = np.sqrt(np.sum(np.square(mu_s - new_mus)))
  mu_s = new_mus
  print("delta : ", delta)

"""
結果をplotしてみる
"""
t_s = []

for x in X:
  delta_0 = np.sum(np.square(x - mu_s[0]))
  delta_1 = np.sum(np.square(x - mu_s[1]))
  delta_2 = np.sum(np.square(x - mu_s[2]))
  min_index = np.argmin(np.array([delta_0, delta_1, delta_2]))
  t_s.append(min_index)

t_s = np.array(t_s)

# plot
plt.scatter(X[t_s == 0][:, 0], X[t_s == 0][:, 1], color="red")
plt.scatter(X[t_s == 1][:, 0], X[t_s == 1][:, 1], color="blue")
plt.scatter(X[t_s == 2][:, 0], X[t_s == 2][:, 1], color="green")
plt.savefig("k-means.png")
    