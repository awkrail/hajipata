# 標準化
# データはirisデータセット
from common import load_iris, plot_scatter
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # データのロード
  setosa, versicolour, virginica = load_iris()
  all_data = np.array(setosa + versicolour + virginica, dtype=np.float32)
  N = all_data.shape[0]

  # 平均と分散の計算
  length, width = all_data[:, 0], all_data[:, 1]
  mu_l, mu_w = np.mean(length), np.mean(width)
  mu = np.array([mu_l, mu_w], dtype=np.float32)
  sigma_ll = (1/N) * np.sum((length - mu_l)**2)
  sigma_lw = (1/N) * np.sum((length - mu_l)*(width - mu_w))
  sigma_ww = (1/N) * np.sum((width - mu_w)**2)

  # normalization
  n_length = (length - mu_l) / np.sqrt(sigma_ll)
  n_width = (width - mu_w) / np.sqrt(sigma_ww)
  
  # plot
  plt.clf()
  plt.scatter(n_length[:50], n_width[:50], marker='.', color='red')
  plt.scatter(n_length[50:100], n_width[50:100], marker='o', color='blue')
  plt.scatter(n_length[100:], n_width[100:], marker='^', color='green')
  plt.title('あやめデータ')
  plt.xlabel('花弁の長さ')
  plt.ylabel('花弁の幅')
  plt.savefig("results/normalized.png")
