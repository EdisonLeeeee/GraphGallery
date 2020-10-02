import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Top_k_features(Layer):
    """
        Top_k_features layer as in:
        [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965)
        Tensorflow 1.x implementation: https://github.com/divelab/lgcn

        `Top_k_features` implements the operation:
        Select the top-K attributes for each node and each attribute dimension.
        And finally the selected attributes will concatenated with the input attribute matrix along last dimension.

        Parameters:
          K: Positive Integer, Number of top elements to look for.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_attrs), (n_nodes, n_nodes)]`.
          The former one is the attribute matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          3-D tensor with shape: `(n_nodes, K+1, n_attrs)`.
    """

    def __init__(self, K, **kwargs):

        super().__init__(**kwargs)
        self.K = K

    def call(self, inputs):

        x, adj = inputs
        if K.is_sparse(adj):
            adj = tf.sparse.to_dense(adj)  # the adjacency matrix will be transformed into dense matrix
        adj = tf.expand_dims(adj, axis=1)  # (N, 1, N)
        x = tf.expand_dims(x, axis=-1)  # (N, F, 1)
        h = adj * x  # (N, F, N)
        h = tf.transpose(h, perm=(2, 1, 0))
        h = tf.math.top_k(h, k=self.K, sorted=True).values
        h = tf.concat([x, h], axis=-1)
        h = tf.transpose(h, perm=(0, 2, 1))
        return h  # (N, K+1, F)

    def get_config(self):
        config = {'K': self.K}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        output_shape = (attributes_shape[0], self.K+1, attributes_shape[1])
        return tf.TensorShape(output_shape)
