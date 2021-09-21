import tensorflow as tf
from tensorflow.keras.layers import Layer
from ..dropout import MixedDropout


class APPNProp(Layer):

    def __init__(self,
                 alpha=0.1,
                 K=10,
                 dropout=0.,
                 **kwargs):

        super().__init__(**kwargs)
        self.alpha = alpha
        self.K = K
        self.dropout = MixedDropout(dropout)

    def call(self, inputs):
        x, adj = inputs
        h = x
        for _ in range(self.K):
            A_drop = self.dropout(adj)
            h = (1 - self.alpha) * tf.sparse.sparse_dense_matmul(A_drop, h) + self.alpha * x
        return h

    def get_config(self):
        config = {'alpha': self.alpha,
                  'K': self.K,
                  'dropout': self.dropout
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)


class PPNProp(Layer):

    def __init__(self,
                 dropout=0.,
                 **kwargs):

        super().__init__(**kwargs)
        self.dropout = MixedDropout(dropout)

    def call(self, inputs):
        x, adj = inputs
        A_drop = self.dropout(adj)
        return A_drop @ x

    def get_config(self):
        config = {'dropout': self.dropout}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)
