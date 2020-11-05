import tensorflow as tf
from tensorflow.keras.layers import Layer


class SGConvolution(Layer):
    """
        Simplifying graph convolution layer as in: 
        [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
        Pytorch implementation: https://github.com/Tiiiger/SGC

        `SGConvolution` implements the operation:
        `output = x @ adj^{order}`
        where `x` is the attribute matrix, `adj` is the adjacency matrix.

        Note:
          This `SGConvolution` layer has NOT any trainable parameters.


        Parameters:
          order: Positive integer, the power of adjacency matrix, i.e., adj^{order}.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_attrs), (n_nodes, n_nodes)]`.
          The former one is the attribute matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          2-D tensor with shape: `(n_nodes, n_attrs)`.       
    """

    def __init__(self, order=1, **kwargs):
        """
        Initialize the order.

        Args:
            self: (todo): write your description
            order: (int): write your description
        """
        super().__init__(**kwargs)
        self.order = order

    def call(self, inputs):
        """
        Call tf.

        Args:
            self: (todo): write your description
            inputs: (dict): write your description
        """
        x, adj = inputs

        for _ in range(self.order):
            x = tf.sparse.sparse_dense_matmul(adj, x)

        return x

    def get_config(self):
        """
        Returns the configurations of configurations

        Args:
            self: (str): write your description
        """
        config = {'order': self.order}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the shape.

        Args:
            self: (todo): write your description
            input_shapes: (list): write your description
        """
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)  # (n_nodes, n_attrs)
