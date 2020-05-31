from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer

import numpy as np
import tensorflow as tf

from graphgallery.utils import to_something


class NodeSequence(Sequence):

    def __init__(self, *args, **kwargs,):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def to_tensor(inputs):
        return to_something.to_tensor(inputs)

    def on_epoch_end(self):
        pass

    def shuffle(self):
        pass


class SqueezedSparseConversion(Layer):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.matrix_shape = shape

    def get_config(self):
        config = {"shape": self.matrix_shape, "dtype": self.dtype}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        return self.matrix_shape

    def call(self, inputs):
        indices, values = inputs
        if indices.dtype != tf.int64:
            indices = tf.cast(indices, tf.int64)

        # Build sparse tensor for the matrix
        output = tf.SparseTensor(
            indices=indices, values=values, dense_shape=self.matrix_shape
        )
        return output
