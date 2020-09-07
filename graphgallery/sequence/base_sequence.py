import numpy as np
import tensorflow as tf


from tensorflow.keras.utils import Sequence as tf_Sequence
from tensorflow.keras.layers import Layer


class Sequence(tf_Sequence):

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
