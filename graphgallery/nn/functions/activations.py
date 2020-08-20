import numpy as np


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""    
    exp_x = np.exp(x - np.max(x))
    return exp_x/exp_x.sum(axis=axis, keepdims=True)