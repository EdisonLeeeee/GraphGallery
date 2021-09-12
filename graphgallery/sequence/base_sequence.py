import numpy as np
import tensorflow as tf


from tensorflow.keras.utils import Sequence as tf_Sequence
from functools import partial

from graphgallery import functional as gf


class Sequence(tf_Sequence):

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', 'cpu')
        escape = kwargs.pop('escape', None)
        super().__init__(*args, **kwargs)
        self.astensor = partial(gf.astensor, device=device, escape=escape)
        self.astensors = partial(gf.astensors, device=device, escape=escape)
        self.device = device

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...

    def _shuffle_batches(self):
        ...
