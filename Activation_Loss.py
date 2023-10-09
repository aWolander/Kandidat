import numpy as np


def loss(y_hat, y, loss_func):
    if loss_func == "L2":
        return loss_L2(y_hat, y)
    if loss_func == "entropy":
        return entropy(y_hat, y)
    raise "Loss function does not exist"


def loss_prim(y_hat, y, loss_func):
    if loss_func == "L2":
        return loss_L2_prim(y_hat, y)
    if loss_func == "entropy":
        return cross_entropy_prim(y_hat, y)
    raise "Loss function derivative does not exist"


def activation(x, activation_func):
    if activation_func == "sigmoid":
        return sigmoid(x)
    if activation_func == "ReLU":
        return ReLU(x)
    if activation_func == "softmax":
        return softmax(x)
    if activation_func == "identity":
        return identity(x)
    raise "Activation function does not exist"


def activation_prim(x, activation_func):
    if activation_func == "sigmoid":
        return sigmoid_prim(x)
    if activation_func == "ReLU":
        return ReLU_prim(x)
    if activation_func == "softmax":
        return softmax_prim(x)
    if activation_func == "identity":
        return identity_prim(x)
    raise "Activation function derivative does not exist"


def identity(x):
    return x


def identity_prim(x):
    return 1


def loss_L2(y_hat, y):
    loss_num = 1 / 2 * (np.linalg.norm(y_hat - y)) ** 2
    return loss_num


def loss_L2_prim(y_hat, y):
    return y_hat - y


def entropy(y_hat, y):
    return -np.log(y_hat[np.argmax(y)])[0]


def cross_entropy_prim(y_hat, y):
    # only works with softmax function, be careful
    return y_hat - y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# leaky ReLU
def ReLU(x):
    return np.where(x >= 0, x, x * 0.1)


def ReLU_prim(x):
    return np.where(x >= 0, 1, 0.1)


def sigmoid_prim(x):
    return sigmoid(x) * (np.ones([np.size(x), 1]) - sigmoid(x))


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def softmax_prim(x):
    softmax_matrix= softmax(x)
    derivative = -np.outer(softmax_matrix, softmax_matrix) + np.diag(softmax_matrix.flatten())
    return derivative
