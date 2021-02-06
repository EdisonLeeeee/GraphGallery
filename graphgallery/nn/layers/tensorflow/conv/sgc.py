import tensorflow as tf
from tensorflow.keras.layers import Layer


class SGConvolution(Layer):
    """
        Simplifying graph convolution layer as in: 
        [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
        Pytorch implementation: https://github.com/Tiiiger/SGC

        `SGConvolution` implements the operation:
        `output = x @ adj^{K}`
        where `x` is the node attribute matrix, `adj` is the adjacency matrix.

        Note:
          This `SGConvolution` layer has NOT any trainable parameters.


        Parameters:
          K: Positive integer, the power of adjacency matrix, i.e., adj^{K}.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(num_nodes, num_node_attrs), (num_nodes, num_nodes)]`.
          The former one is the node attribute matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(num_nodes, num_node_attrs)`.       
    """

    def __init__(self, K=1, **kwargs):
        super().__init__(**kwargs)
        self.K = K

    def call(self, inputs):
        x, adj = inputs

        for _ in range(self.K):
            x = tf.sparse.sparse_dense_matmul(adj, x)

        return x

    def get_config(self):
        config = {'K': self.K}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)  # (num_nodes, num_node_attrs)
