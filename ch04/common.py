# 4章の共通の処理のまとめ
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_iris():
  iris = datasets.load_iris()
  petal = iris['data'][:, 2:]
  target = iris['target']
  setosa = []
  versicolour = []
  virginica = []

  for i, y in enumerate(target):
    if y == 0:
      setosa.append(petal[i])
    elif y == 1:
      versicolour.append(petal[i])
    elif y == 2:
      virginica.append(petal[i])
  
  return setosa, versicolour, virginica

def plot_scatter(flower, marker, color):
  petal_length = np.array([f[0] for f in flower], dtype=np.float32)
  petal_width = np.array([f[1] for f in flower], dtype=np.float32)
  plt.scatter(petal_length, petal_width, marker=marker, color=color)