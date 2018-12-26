import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


def add_ones(x):
    one = np.ones((x.shape[0], 1))
    return np.concatenate((one, x), axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def random_initialize(a, b):
    epsilon = sqrt(6) / (sqrt(a) + sqrt(b))
    return np.random.rand(a, b+1)*2*epsilon - epsilon


def nn_cost_function(X, y, theta, alpha, num_features, num_hidden_layer, num_labels):
    m = X.shape[0]
    J = 0
    # Reshape theta
    Theta1 = np.reshape(theta[:num_hidden_layer*(num_features+1)], (num_hidden_layer, num_features+1))
    Theta2 = np.reshape(theta[num_hidden_layer*(num_features+1):], (num_labels, num_hidden_layer+1))
    theta1_grad = np.zeros_like(Theta1)
    theta2_grad = np.zeros_like(Theta2)

    for i in range(m):
        z2 = np.array([np.dot(Theta1, X[0, :].T)])
        a2 = sigmoid(z2)
        a2 = add_ones(a2)
        output = sigmoid(np.dot(Theta2, a2.T))
        J -= 1/m * (np.dot(np.array([y[:, i]]), np.log(output))
                    + np.dot(np.array([1 - y[:, i]]), np.log(1 - output)))

        # Delta for gradient
        zero = np.zeros((1, 1))
        delta3 = output - np.array([y[:, 1]]).T
        delta2 = np.dot(Theta2.T, delta3) * np.concatenate((zero, sigmoid_gradient(z2)), axis=1).T
        delta2 = np.array([delta2[1:, 0]]).T

        theta1_grad += alpha/m * np.dot(delta2, np.array([X[i, :]]))
        theta2_grad += alpha/m * np.dot(delta3, a2)

    theta_grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))

    return J[0][0], theta_grad


def predict(Xval, theta1, theta2):
    a2 = add_ones(sigmoid(np.dot(Xval, theta1.T)))
    output = sigmoid(np.dot(theta2, a2.T))
    prediction = np.zeros_like(output)
    max = np.argmax(output, axis=0)
    return prediction


# def check_grad(X, y, theta, alpha, num_features, num_hidden_layer, num_labels):
#     epsilon = 1e-4 * np.ones_like(theta)
#     cost1, _ = nn_cost_function(X, y, theta+epsilon, alpha, num_features, num_hidden_layer, num_labels)
#     cost2, _ = nn_cost_function(X, y, theta-epsilon, alpha, num_features, num_hidden_layer, num_labels)
#     cost, _ = nn_cost_function(X, y, theta, alpha, num_features, num_hidden_layer, num_labels)
#     print(abs((cost1 - cost2) / (2*epsilon) - cost))
#     if abs((cost1 - cost2) / (2*epsilon) - cost) < 1e-4:
#         return True
#     else:
#         return False


# Load data
df = pd.read_csv("Data/train.csv")
X = np.array([df['bone_length'], df['rotting_flesh'], df['hair_length'], df['has_soul']]).T
y = np.array([df['type']]).T
X = add_ones(X)

# Some useful variables
m = X.shape[0]
n = X.shape[1]
num_features = 4
num_hidden_layer = 5
num_labels = 3

alpha = 0.003
iterations = 300

# Modify y
y_temp = y
y = np.zeros([num_labels, m])
for i in range(m):
    if y_temp[i][0] == 'Ghoul':
        y[0][i] = 1
        y_temp[i][0] = 0
        continue
    if y_temp[i][0] == 'Goblin':
        y[1][i] = 1
        y_temp[i][0] = 1
        continue
    if y_temp[i][0] == 'Ghost':
        y[2][i] = 1
        y_temp[i][0] = 2
        continue

# Split train and test data
X_train = X[0:300, :]
y_train = y[0:300, :]
X_test = X[300:-1, :]
y_test = y[300:-1, :]

theta1 = random_initialize(num_hidden_layer, num_features)
theta2 = random_initialize(num_labels, num_hidden_layer)
theta = np.concatenate((theta1.ravel(), theta2.ravel()))

# Computing
J_list = []
print("Compute cost...")
for i in range(iterations):
    J, theta_grad = nn_cost_function(X, y, theta, alpha, num_features, num_hidden_layer, num_labels)
    J_list.append(J)
    theta -= theta_grad

theta1 = np.reshape(theta[:num_hidden_layer*(num_features+1)], (num_hidden_layer, num_features+1))
theta2 = np.reshape(theta[num_hidden_layer*(num_features+1):], (num_labels, num_hidden_layer+1))

print(theta1, theta2)
prediction = predict(X_test, theta1, theta2)

J_list = np.array(J_list)
iter = np.linspace(1, 300, 300)
print(J_list.shape)
# check = check_grad(X, y, theta, alpha, num_features, num_hidden_layer, num_labels)
# print(check)

# Visualizing
print("Visualizing...")

plt.subplot(2, 2, 1)
plt.plot(X[y_temp[:, 0] == 0][:, 1], X[y_temp[:, 0] == 0][:, 2], 'ro')
plt.plot(X[y_temp[:, 0] == 1][:, 1], X[y_temp[:, 0] == 1][:, 2], 'g^')
plt.plot(X[y_temp[:, 0] == 2][:, 1], X[y_temp[:, 0] == 2][:, 2], 'bs')
plt.xlabel("Bone length")
plt.ylabel("Rotting flesh")
plt.axis([0, 1, 0, 1])

plt.subplot(2, 2, 2)
plt.plot(iter, J_list)
plt.axis([0, 300, 0, 5])
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.show()
