import numpy as np


# def normalize_edges(edge_index, edge_weights, degree):
#     row, col = indices.T
#     inv_degree = tf.pow(degree, -0.5)
#     normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(inv_degree, col)
#     return normed_weights