# 白色化
# データはアイリスデータセット
from common import load_iris
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # 必要なデータのロード
  setosa, versicolour, virginica = load_iris()
  all_data = np.array(setosa + versicolour + virginica, dtype=np.float32)
  N = all_data.shape[0]

  # 平均と分散を計算する
  length, width = all_data[:, 0], all_data[:, 1]
  mu_l, mu_w = np.mean(length), np.mean(width)
  mu = np.array([mu_l, mu_w], dtype=np.float32)
  sigma_ll = (1/N) * np.sum((length - mu_l)**2)
  sigma_lw = (1/N) * np.sum((length - mu_l)*(width - mu_w))
  sigma_ww = (1/N) * np.sum((width - mu_w)**2)
  sigma = np.array([[sigma_ll, sigma_lw], [sigma_lw, sigma_ww]], dtype=np.float32)

  # 固有値と固有ベクトルの計算
  lambdas, S = np.linalg.eig(sigma)
  # y = np.dot(S.T, all_data.T).T

  # 固有値の行列
  A = np.zeros((2, 2))
  A[0][0] = np.sqrt(lambdas[0])
  A[1][1] = np.sqrt(lambdas[1])
  A = np.linalg.inv(A)

  # 白色化
  y = np.dot(S.T, (all_data - mu).T)
  u = np.dot(A, y).T

  # x軸成分とy軸成分に分解
  n_length = u[:, 0]
  n_width = u[:, 1]

  # plot
  plt.clf()
  plt.scatter(n_length[:50], n_width[:50], marker='.', color='red')
  plt.scatter(n_length[50:100], n_width[50:100], marker='o', color='blue')
  plt.scatter(n_length[100:], n_width[100:], marker='^', color='green')
  plt.xlim([-3.0, 3.0])
  plt.ylim([-2.0, 3.0])
  plt.title('あやめデータ')
  plt.xlabel('花弁の長さ')
  plt.ylabel('花弁の幅')
  plt.savefig("results/whitening.png")



