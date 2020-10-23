import numpy as np
import tensorflow as tf


from tensorflow.keras.utils import Sequence as tf_Sequence
from torch.nn import Module
from functools import partial

from graphgallery import transforms as T

class Sequence(tf_Sequence):

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', 'cpu')
        super().__init__(*args, **kwargs)
        self.astensor = partial(T.astensor, device=device)
        self.astensors = partial(T.astensors, device=device)
        self.device = device
        

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def on_epoch_end(self):
        ...

    def _shuffle_batches(self):
        ...
        
