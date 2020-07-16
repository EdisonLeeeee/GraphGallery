import numpy as np
import tensorflow as tf


from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer


class NodeSequence(Sequence):

    def __init__(self, *args, **kwargs,):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def on_epoch_end(self):
        pass

    def shuffle(self):
        pass
