import numpy as np


def step_function(x, binary=True):
    if x >= 0:
        return 1
    if binary:
        return 0
    return -1


def linear_activation_function(x):
    return x


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def RluU_function(x):
    return max(0, x)


def leaky_RluU_function(x):
    if x > 0:
        return x
    return 0.01 * x


def softmax_function(x):
    exp_values = np.exp(x)
    return exp_values / exp_values.sum(axis=0)
