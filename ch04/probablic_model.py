# 確率モデル
# gluとbmiの特徴をプロットし, 正規分布を用いた線形識別関数を書く
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  pima_df = pd.read_csv("pimatr.csv")
  pima_df = pima_df[["glu", "bmi", "type"]]
  yes_df = pima_df[pima_df["type"] == "Yes"]
  no_df = pima_df[pima_df["type"] == "No"]

  # glu, bmiをプロットする
  yes_data = yes_df[["glu", "bmi"]].get_values()
  no_data = no_df[["glu", "bmi"]].get_values()
  plt.scatter(yes_data[:, 0], yes_data[:, 1], marker='o', color='red')
  plt.scatter(no_data[:, 0], no_data[:, 1], marker='o', color='blue')
  plt.title('Pimatr データセット')
  plt.xlabel('glu')
  plt.ylabel("bmi")
  plt.savefig("results/pimatr/pimatr.png")

  # 事前確率を求める
  p_c1 = 132 / 200
  p_c2 = 68 / 200

  # mu1, mu2, sigma1, sigma2を求める
  mu1 = np.mean(no_data, axis=0)
  mu2 = np.mean(yes_data, axis=0)
  sigma1 = np.cov(no_data.T)
  sigma2 = np.cov(yes_data.T)

  inv_sigma1 = np.linalg.inv(sigma1)
  inv_sigma2 = np.linalg.inv(sigma2)

  # S, c, Fを計算する
  S = inv_sigma1 - inv_sigma2
  c_t = np.dot(mu2.T, inv_sigma2) - np.dot(mu1.T, inv_sigma1)
  F = np.dot(np.dot(mu1.T, inv_sigma1), mu1) - np.dot(np.dot(mu2.T, inv_sigma2), mu2)
  F += np.log(np.linalg.det(sigma1) / np.linalg.det(sigma2))
  F -= 2 * np.log(p_c1 / p_c2)

  # TODO : 等高線を描いてグラフをplotする
  def f(X, Y):
    x = np.array([X, Y], dtype=np.float32)
    import ipdb; ipdb.set_trace()
    return np.dot(np.dot(x.T, S), x) + 2*np.dot(c_t, x) + F
  
  n = 256
  plt.clf()

  # glu, bmiをプロットする
  yes_data = yes_df[["glu", "bmi"]].get_values()
  no_data = no_df[["glu", "bmi"]].get_values()
  plt.scatter(yes_data[:, 0], yes_data[:, 1], marker='o', color='red')
  plt.scatter(no_data[:, 0], no_data[:, 1], marker='o', color='blue')
  plt.title('Pimatr データセット')
  plt.xlabel('glu')
  plt.ylabel("bmi")
  plt.show()




  