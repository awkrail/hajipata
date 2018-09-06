# 無相関化
# データはアイリスデータセット
from common import load_iris
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  setosa, versicolour, virginica = load_iris()
  
  setosa_N = len(setosa)
  versicolour_N = len(versicolour)
  virginica_N = len(virginica)

  all_data = np.array(setosa + versicolour + virginica, dtype=np.float32)
  N = all_data.shape[0]

  length, width = all_data[:, 0], all_data[:, 1]
  mu_l, mu_w = np.mean(length), np.mean(width)
  mu = np.array([mu_l, mu_w], dtype=np.float32)
  sigma_ll = (1/N) * np.sum((length - mu_l)**2)
  sigma_lw = (1/N) * np.sum((length - mu_l)*(width - mu_w))
  sigma_ww = (1/N) * np.sum((width - mu_w)**2)

  sigma = np.array([[sigma_ll, sigma_lw], [sigma_lw, sigma_ww]], dtype=np.float32)
  result = np.linalg.eig(sigma)
  for i in range(len(result[1])):
    result[1][i] = result[1][i] / np.linalg.norm(result[1][i])
  import ipdb; ipdb.set_trace()
  lambdas, S_t = result[0], result[1].T
  y = all_data @ S_t

  # divide y with length, and width
  n_length = y[:, 0]
  n_width = y[:, 1]

  # plot
  setosa = np.concatenate((n_length[:setosa_N], n_width[:setosa_N])).reshape(setosa_N, 2)
  versicolour = np.concatenate((n_length[setosa_N:setosa_N+versicolour_N], n_width[setosa_N:setosa_N+versicolour_N])).reshape(versicolour_N, 2)
  virginica = np.concatenate((n_length[setosa_N+versicolour_N:], n_width[setosa_N+versicolour_N:])).reshape(virginica_N, 2)
  
  # plot
  plt.clf()
  plt.scatter(setosa[:, 0], setosa[:, 1], marker='.', color='red')
  plt.scatter(versicolour[:, 0], versicolour[:, 1], marker='o', color='blue')
  plt.scatter(virginica[:, 0], virginica[:, 1], marker='^', color='green')
  plt.title('あやめデータ')
  plt.xlabel('花弁の長さ')
  plt.ylabel('花弁の幅')
  plt.savefig("results/decorrelated.png")
