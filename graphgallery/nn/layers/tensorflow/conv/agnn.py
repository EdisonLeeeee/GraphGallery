from tensorflow.keras.layers import Layer
from tensorflow.keras import constraints, initializers, regularizers

import tensorflow as tf


def cosine_similarity(A, B):
    inner_product = tf.reduce_sum(A * B, axis=1)
    C = inner_product / (tf.norm(A, 2, 1) * tf.norm(B, 2, 1) + 1e-7)
    return C


class AGNNConv(Layer):

    def __init__(self, trainable=True,
                 initializer='ones',
                 regularizer=None,
                 constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.trainable = trainable
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    def build(self, input_shapes):
        if self.trainable:
            self.beta = self.add_weight(shape=(1,),
                                        initializer=self.initializer,
                                        name='beta',
                                        regularizer=self.regularizer,
                                        constraint=self.constraint)
        else:
            self.beta = tf.ones(shape=(1,))
        super().build(input_shapes)

    def call(self, inputs):
        x, adj = inputs
        row, col = tf.unstack(adj.indices, axis=1)
        A = tf.gather(x, row)
        B = tf.gather(x, col)
        sim = self.beta * cosine_similarity(A, B)
        adj = tf.sparse.SparseTensor(adj.indices, sim, dense_shape=adj.dense_shape)
        P = tf.sparse.softmax(adj)
        output = tf.sparse.sparse_dense_matmul(P, x)
        return output

    def get_config(self):
        config = {'trainable': self.trainable,
                  }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        attributes_shape = input_shapes[0]
        return tf.TensorShape(attributes_shape)  # (num_nodes, output_dim)
