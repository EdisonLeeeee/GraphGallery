import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Top_k_features(Layer):
    """
        Top_k_features layer as in:
        [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965)
        Tensorflow 1.x implementation: https://github.com/divelab/lgcn

        `Top_k_features` implements the operation:
        Select the top-k features for each node and each feature dimension.
        And finally the selected features will concatenated with the input feature matrix along last dimension.

        Arguments:
          k: Positive Integer, Number of top elements to look for.

        Input shape:
          tuple/list with two 2-D tensor: Tensor `x` and SparseTensor `adj`: `[(n_nodes, n_features), (n_nodes, n_nodes)]`.
          The former one is the feature matrix (Tensor) and the other is adjacency matrix (SparseTensor).

        Output shape:
          3-D tensor with shape: `(n_nodes, k+1, n_features)`.
    """

    def __init__(self, k, **kwargs):

        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):

        x, adj = inputs
        if K.is_sparse(adj):
            adj = tf.sparse.to_dense(adj) # the adjacency matrix will be transformed into dense matrix
        adj = tf.expand_dims(adj, axis=1)  # (N, 1, N)
        x = tf.expand_dims(x, axis=-1)  # (N, F, 1)
        h = adj * x  # (N, F, N)
        h = tf.transpose(h, perm=(2, 1, 0))
        h = tf.math.top_k(h, k=self.k, sorted=True).values
        h = tf.concat([x, h], axis=-1)
        h = tf.transpose(h, perm=(0, 2, 1))
        return h  # (N, k+1, F)

    def get_config(self):
        config = {'k': self.k}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.k+1, features_shape[1])
        return tf.TensorShape(output_shape)
