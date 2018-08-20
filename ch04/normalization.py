# 標準化
# データはirisデータセット
from common import load_iris, plot
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  setosa, versicolour, virginica = load_iris()
  plot(setosa, marker='.', color='red')
  plot(versicolour, marker='o', color='blue')
  plot(virginica, marker='^', color='green')
  plt.title('あやめデータ')
  plt.xlabel('花弁の長さ')
  plt.ylabel('花弁の幅')
  plt.savefig("results/original.png")
