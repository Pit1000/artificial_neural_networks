import numpy as np
import matplotlib.pyplot as plt  # library for plots
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
from sklearn.datasets import make_classification  # data generation, sklearn - very useful library for machine learning

def initialize():
    # We randomly initialize W values from a normal distribution with mean 0 and standard deviation 0.1, set b values to 0
    W1 = np.random.normal(0, 0.1, size=(4, 3))
    b1 = np.zeros(shape=(4, 1))
    W2 = np.random.normal(0, 0.1, size=(1, 4))
    b2 = np.zeros(shape=(1, 1))
    return W1, b1, W2, b2

# Activation functions
def sigmoid(Z):
    # sigmoid activation function for matrix Z
    return 1 / (1 + np.exp(-Z))

def leaky_relu(Z):
    # leaky relu activation function for matrix Z (leaky_relu(Z) = max(0.01 * Z, Z))
    return np.maximum(0.01 * Z, Z)

def leaky_relu_grad(Z):
    # derivative of leaky relu function for matrix Z
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    return np.where(Z < 0, 0.01, 1)

def forward(X, W1, W2, b1, b2):
    # determine the network output and values in hidden layers
    Z1 = np.dot(W1, X) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    Y_hat = A2
    return Z1, A1, Z2, A2, Y_hat

def J(Y, Y_hat):
    # binary cross entropy
    # please use natural logarithm (np.log)
    return -1 / np.size(Y) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def calculate_gradients(Y, A2, A1, Z1, W2, W1, b2, b1, X, m):
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, np.transpose(A1))
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(W2), dZ2) * leaky_relu_grad(Z1)
    dW1 = np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW2, db2, dW1, db1

def update(W2, W1, b2, b1, dW2, dW1, db2, db1, alpha=0.001):
    W2 = W2 - (alpha * dW2)
    W1 = W1 - (alpha * dW1)
    b2 = b2 - (alpha * db2)
    b1 = b1 - (alpha * db1)
    return W2, W1, b2, b1

def accuracy(Y, Y_hat):
    return 100 * np.sum(np.where(np.where(Y_hat > 0.5, 1, 0) == Y, 1, 0)) / np.size(Y)

m = 300
data = make_classification(n_samples=m, n_features=3, n_informative=3, n_redundant=0, n_classes=2, random_state=0)

X = data[0]
Y = data[1]
X.shape
Y.shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y.transpose())
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
fig.colorbar(cax)
plt.show()

# weight initialization
np.random.seed(0)
W1, b1, W2, b2 = initialize()

print('W1 = ', W1)
print('b1 = ', b1)
print('W2 = ', W2)
print('b2 = ', b2)

assert(W1.shape == (4, 3))
assert(b1.shape == (4, 1))
assert(W2.shape == (1, 4))

X.shape
Xt = X.transpose()
Xt.shape
Y = Y.reshape(1, m)

Z1, A1, Z2, A2, Y_hat = forward(Xt, W1, W2, b1, b2)

assert(Z1.shape == (4, m))
assert(Z2.shape == (1, m))
assert(Y_hat.shape == (1, m))

dW2, db2, dW1, db1 = calculate_gradients(Y, A2, A1, Z1, W2, W1, b2, b1, Xt, m)

W2, W1, b2, b1 = update(W2, W1, b2, b1, dW2, dW1, db2, db1)
assert(W2.shape == (1, 4))

# weight initialization
W1, b1, W2, b2 = initialize()
# list to save cost function values in subsequent learning steps (initially empty)
J_history = []
# list to save accuracy in subsequent steps (initially empty)
acc_history = []
for i in range(100000):
    Z1, A1, Z2, A2, Y_hat = forward(Xt, W1, W2, b1, b2)
    J_history.append(J(Y, Y_hat))
    acc_history.append(accuracy(Y, Y_hat))
    dW2, db2, dW1, db1 = calculate_gradients(Y, A2, A1, Z1, W2, W1, b2, b1, Xt, m)
    W2, W1, b2, b1 = update(W2, W1, b2, b1, dW2, dW1, db2, db1, alpha=0.001)

plt.plot(J_history)
plt.title('Cost Function vs. Iterations')
plt.show()

plt.plot(acc_history)
plt.title('Accuracy vs. Iterations')
plt.show()

# final accuracy
acc_history[-1]

# point to test
x_test1 = np.array([[3, 3, 0]])

# new point marked with a red dot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y.reshape(m,))
ax.scatter(x_test1[0][0], x_test1[0][1], x_test1[0][2], c='r')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
fig.colorbar(cax)
plt.show()

# prediction close to 1 indicates a yellow dot, close to 0 a purple dot
Z1, A1, Z2, A2, Y_hat = forward(x_test1.transpose(), W1, W2, b1, b2)
print(Y_hat)
