import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random

class C_SVM:
  def __init__(self, X, t, C=1.0, eps=0.01, tol=0.01):
    self.a = np.zeros(X.shape[0])
    self.X = X
    self.t = t
    self.E = t * (-1)
    self.C = C
    self.b = 0
    self.eps = eps
    self.tol = tol
    self.N = X.shape[0]
    self.dim = X.shape[1]
  
  def kernel(self, x1, x2, delta=1.0):
    tmp = 0
    for i in range(self.dim):
      tmp += (x1[i] - x2[i])*(x1[i] - x2[i])
    return np.exp(-tmp/2.0*delta*delta)
  
  def predict(self, x):
    tmp = 0
    for i in range(self.N):
      tmp += self.a[i] * self.t[i] * self.kernel(x, self.X[i])
    return tmp - self.b
  
  def takeStep(self, i1, i2):
    if i1 == i2:
      return 0

    alph1 = self.a[i1]
    alph2 = self.a[i2]
    y1 = self.t[i1]
    y2 = self.t[i2]
    E1 = self.E[i1]
    E2 = self.E[i2]
    s = y1 * y2

    # L, Hを計算する
    L, H = 0, 0
    if y1 != y2:
      L = max(0.0, alph2-alph1)
      H = min(self.C, self.C+alph2-alph1)
    else:
      L = max(0.0, alph1+alph2-self.C)
      H = min(self.C, alph1+alph2)
    if L == H:
      return 0
    
    k11 = self.kernel(self.X[i1], self.X[i1])
    k12 = self.kernel(self.X[i1], self.X[i2])
    k22 = self.kernel(self.X[i2], self.X[i2])
    eta = 2*k12-k11-k22

    a1, a2 = 0, 0
    """
    etaは逆行列式を正負反転したものである
    グラム行列が正則である => 半正定値であるかどうかで場合分け
    """
    if eta < 0:
      a2 = alph2 - y2 * (E1 - E2)/eta
      if a2 < L:
        a2 = L
      elif a2 > H:
        a2 = H
    else:
      """
      ここの処理 : 分かってない
      """
      print("not positive semi-definite")
      a1 = self.a[i1]
      a2 = self.a[i2]
      v1 = self.predict(self.X[i1]) - self.b - y1*a1*k11 - y1*a2*k12
      v2 = self.predict(self.X[i2]) - self.b - y1*a1*k12 - y2*a2*k22
      Wconst = 0
      
      for i in range(self.N):
        if i != i1 and i != i2:
          Wconst += self.a[i1]
      
      for i in range(self.N):
        for j in range(self.N):
          if i != i1 and i != i2 and j != i1 and j != i2:
            Wconst += self.t[i]*self.t[j]*self.kernel(self.X[i], self.X[j])*self.a[i]*self.a[j]/2.0
      a2 = L
      a1 = y1*self.a[i1] + y2*self.a[i2] - y2*L
      Lobj = a1+a2-k11*a1*a1/2.0-k22*a2*a2/2.0-s*k12*a1*a2/2.0 -y1*a1*v1-y2*a2*v2+Wconst

      a2 = H
      a1 = y1*self.a[i1]+y2*self.a[i2]-y2*H
      Hobj = a1+a2-k11*a1*a1/2.0-k22*a2*a2/2.0-s*k12*a1*a2/2.0 -y1*a1*v1-y2*a2*v2+Wconst

      if Lobj > Hobj + self.eps:
        a2 = L
      elif(Lobj < Hobj - self.eps):
        a2 = H
      else:
        a2 = alph2
    
    """
    更新があまりに小さい場合は更新を行わない
    """
    if a2 < 1e-8:
      a2 = 0
    elif a2 > self.C-1e-8:
      a2 = self.C
    if abs(a2 - alph2) < self.eps*(a2+alph2+self.eps):
      return 0
    
    """
    バイアスと2点目の更新処理
    """
    a1 = alph1+s*(alph2-a2)

    b_old = self.b
    b1 = E1 + y1*(a1 - self.a[i1])*k11 + y2*(a2 - self.a[i2])*k12 + self.b
    b2 = E2 + y1*(a1 - self.a[i1])*k12 + y2*(a2 - self.a[i2])*k22 + self.b
    if b1 == b2:
      self.b = b1
    else:
      self.b = (b1 + b2) / 2.0

    """
    E[i]の更新(元論文 : 12.2.3 Error Cacheより)
    """
    da1 = a1 - self.a[i1]
    da2 = a2 - self.a[i2]
    for i in range(self.N):
      self.E[i] = self.E[i] + y1*da1*self.kernel(self.X[i1], self.X[i]) + y2*da2*self.kernel(self.X[i2], self.X[i]) + (b_old - self.b)

    self.a[i1] = a1
    self.a[i2] = a2

    return 1

  
  def examineExample(self, i2):
    y2 = self.t[i2]
    alph2 = self.a[i2]
    E2 = self.E[i2]
    r2 = E2 * y2
    i1 = 0

    if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):
      number = 0
      """
      2つ以上, 0<a<Cのものがある場合は, その中で最大のものを探す。 
      self.a[i2]は違反することが確定しているのでもう一つを探す
      """
      for i in range(self.N):
        if self.a[i] != 0 or self.a[i] != self.C:
          number += 1
      if number > 1:
        max_value = 0
        for i in range(self.N):
          if abs(self.E[i] - E2) > max_value:
            max_value = abs(self.E[i] - E2)
            i1 = i
        if self.takeStep(i1, i2):
          return 1
      i1 = random.randint(0, N-1)
      if self.takeStep(i1, i2):
        return 1
    return 0
  
  def train(self):
    threshold = 0
    numChanged = 0
    examineAll = 1

    while numChanged > 0 or examineAll == 1:
      numChanged = 0
      if examineAll == 1:
        for i in range(self.N):
          numChanged += self.examineExample(i)
      else:
        for i in range(self.N):
          if self.a[i] != 0 and self.a[i] != self.C:
            numChanged += self.examineExample(i)

      if examineAll == 1:
        examineAll = 0
      elif numChanged == 0:
        examineAll = 1


if __name__ == "__main__":
  # アイリスデータセットのロード
  iris = load_iris()
  X = iris.data
  t = iris.target
  N = 100
  X = X[:N, 2:]
  t = t[:N]
  for i in range(N):
    if t[i] == 0:
      t[i] = -1
  
  X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state=42)

  c_svm = C_SVM(X_train, t_train)
  c_svm.train()

  correct = 0

  for i in range(X_test.shape[0]):
    if(c_svm.predict(X_test[i]) > 0):
      if(t_test[i] == 1):
        correct += 1
    else:
      if(t_test[i] == -1):
        correct += 1
  
  print("accuracy : ", correct / X_test.shape[0])