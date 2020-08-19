import numpy as np


def softmax(x, axis=-1):
    exp_x = np.exp(x)
    return exp_x/exp_x.sum(axis=axis, keepdims=True)