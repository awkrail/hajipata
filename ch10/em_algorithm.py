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

k_num = 3
pi_s = np.random.randn(3)
mu_s = np.random.randn(3, 2)
sigma_s = np.random.randn(3, 2, 2)

delta = 100000
eps = 1.0

def calculate_E_step(X, pi_s, mu_s, sigma_s):
  z_ik = np.zeros((X.shape[0], k_num))
  for x_i in X:
    pass
    """
    TODO : z_ikの期待値をx_iを基に求める
    """

while delta > eps:
  z_ik = calculate_E_step(X, pi_s, mu_s, sigma_s)

import ipdb; ipdb.set_trace()