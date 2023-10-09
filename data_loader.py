from keras.datasets import mnist
import tensorflow as tf
import numpy as np


def to_vec(int):
    y_vec = np.zeros([10, 1])
    y_vec[int] = 1
    return y_vec


def load_MNIST():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y_vec = []
    test_y_vec = []
    ftrain_X = [None] * 60000
    ftest_X = [None] * 10000
    # I want images to be of the right dimensions. Best to do that here instead of every time i run forwardpop
    for number in train_y:
        train_y_vec.append(to_vec(number))
    for number in test_y:
        test_y_vec.append(to_vec(number))
    for i in range(len(train_X)):
        ftrain_X[i] = np.atleast_2d(train_X[i].ravel()).T / 255
    for i in range(len(test_X)):
        ftest_X[i] = np.atleast_2d(test_X[i].ravel()).T / 255
    return [ftrain_X, train_y_vec, ftest_X, test_y_vec]


def load_CIFAR10():
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar10.load_data()
    train_y_vec = []
    test_y_vec = []
    ftrain_X = [None] * 50000
    ftest_X = [None] * 10000
    for label in train_y:
        train_y_vec.append(to_vec(label))
    for label in test_y:
        test_y_vec.append(to_vec(label))
    for i in range(len(train_X)):
        ftrain_X[i] = np.atleast_2d(train_X[i].ravel()).T / 255
    for i in range(len(test_X)):
        ftest_X[i] = np.atleast_2d(test_X[i].ravel()).T / 255
    return [ftrain_X, train_y_vec, ftest_X, test_y_vec]
