from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer

import numpy as np
import tensorflow as tf

from graphgallery.utils import conversion


class NodeSequence(Sequence):

    def __init__(self, *args, **kwargs,):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def to_tensor(inputs):
        return conversion.to_tensor(inputs)

    def on_epoch_end(self):
        pass

    def shuffle(self):
        pass
