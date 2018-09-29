# 多層レイヤパーセプトロン(7.1)
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# load data
mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'])
X_train, y_train = X_train[:100], y_train[:100]
X_test, y_test = X_test[:50], y_test[:50]

# hyper parameters
input_size, input_num = X_train.shape[1], X_train.shape[0]
hidden_size = 1000
output_size = 10
epochs = 1000
learning_lr = 0.01
W_1 = np.random.randn(input_size, hidden_size)
W_2 = np.random.randn(hidden_size, output_size)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  c = np.max(x)
  return np.exp(x - c) / np.sum(np.exp(x - c))

def cross_entropy(y, t):
  return (-1) * np.log(y[int(t)] + 1e-5)


for epoch in range(epochs):
  epoch_loss = 0

  for x, y in zip(X_train, y_train):
    # forward
    h = sigmoid(np.dot(x, W_1))
    o = softmax(np.dot(h, W_2))

    # calculate loss
    loss = cross_entropy(o, y)

    # backward the loss
    grad_y_pred = o - y
    grad_w2 = np.dot(h.reshape(hidden_size, 1), grad_y_pred.reshape(1, output_size))
    grad_h_pred = np.dot(grad_y_pred, W_2.T)
    grad_sig_h_pred = h * (1 - h) * grad_h_pred
    grad_w1 = np.dot(x.reshape(input_size, 1), grad_sig_h_pred.reshape(1, hidden_size))

    # update parameters
    W_1 -= learning_lr * grad_w1
    W_2 -= learning_lr * grad_w2

    epoch_loss += loss

  print("epoch : ", epoch, " loss : ", epoch_loss / input_num)


# predict
correct_count = 0
test_num = X_test.shape[0]

for x, t in zip(X_test, y_test):
  # forward with W1, W2
  # forward
  h = sigmoid(np.dot(x, W_1))
  o = softmax(np.dot(h, W_2))
  pred_index = np.argmax(o)

  if pred_index == t:
    correct_count += 1

print("test_num : ", test_num)
print("accuracy : ", (correct_count / test_num)*100)