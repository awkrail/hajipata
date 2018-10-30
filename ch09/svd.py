import numpy as np

# 分解する行列
c = np.array([[1,1,1],[1,0,0,],[1,0,0],[0,1,1],[0,1,0],[0,0,1]])

"""
特異値行列Σ(sigma) → 右特異行列V(v) → 左特異行列U(u) の順に求める
"""

# C^TCの固有値と固有ベクトルの計算
ctc = np.dot(c.T, c)
eigen_values, eigen_vectors = np.linalg.eig(ctc)

# 特異値の計算
singular_values = np.sqrt(eigen_values)
singular_index = np.argsort(singular_values)[::-1]

# 特異値行列の計算
sigma = np.diag(singular_values[singular_index])

# 右特異行列の計算
v = eigen_vectors[:,singular_index]

# 左特異行列の計算
u = np.array([np.dot(c, v[:,i]) / sigma.diagonal()[i] for i in range(len(sigma.diagonal()))]).T
v = v.T

# 行列の低ランク近似
u_kin = u[:, :2]
sigma_kin = sigma[:2, :2]
v_kin = v[:2]

C_k = np.dot(u_kin, sigma_kin).dot(v_kin)

# 元と比較する
print(c)
print("========")
print(C_k)