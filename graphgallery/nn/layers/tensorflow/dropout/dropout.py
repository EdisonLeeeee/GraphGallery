import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Dropout


class SparseDropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, training=None):
        if training is None:
            training = K.learning_phase()

        if self.p and training:
            x = tf.nn.dropout(x.values, self.p)
            return tf.SparseTensor(x.indices, x, x.dense_shape)
        return x


class MixedDropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.dense_dropout = Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, x):
        if K.is_sparse(x):
            return self.sparse_dropout(x)
        else:
            return self.dense_dropout(x)
