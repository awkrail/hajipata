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

# k-means
k_num = 3
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
  for x in X_train:
    dist_0 = np.sum(np.square(x  - mu_s[0]))
    dist_1 = np.sum(np.square(x - mu_s[1]))
    dist_2 = np.sum(np.square(x - mu_s[2]))
    min_index = np.argmin(np.array([dist_0, dist_1, dist_2]))
    all_array[min_index].append(x)
  
  """
  muを更新する
  この時に更新量をdeltaに入れておく
  """


    