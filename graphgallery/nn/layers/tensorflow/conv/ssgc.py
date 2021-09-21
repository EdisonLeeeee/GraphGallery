import tensorflow as tf
from tensorflow.keras.layers import Layer


class SSGConv(Layer):
    """
        Simple spectral graph convolution layer as in: 
        [Simple Spectral Graph Convolution](https://openreview.net/forum?id=CYO5T-YjWZV)
        Pytorch implementation: https://github.com/allenhaozhu/SSGC   

        Note:
          This `SSGConv` layer has NOT any trainable parameters.

        Parameters:
          K: Positive integer, the power of adjacency matrix, i.e., adj^{K}.
          alpha: float

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(num_nodes, num_node_attrs), (num_nodes, num_nodes)]`.
          The former one is the node attribute matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(num_nodes, num_node_attrs)`.       
    """

    def __init__(self, K=16, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha

    def call(self, inputs):
        x, adj = inputs
        x_in = x
        x_out = tf.zeros_like(x)
        for _ in range(self.K):
            x = tf.sparse.sparse_dense_matmul(adj, x)
            x_out += (1 - self.alpha) * x
        x_out /= self.K
        x_out += self.alpha * x_in
        return x_out

    def get_config(self):
        config = {'K': self.K, 'alpha': self.alpha}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)  # (num_nodes, num_node_attrs)
