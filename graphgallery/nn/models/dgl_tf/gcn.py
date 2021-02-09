import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from dgl.nn.tensorflow import GraphConv
from tensorflow.keras import activations
from graphgallery.nn.models import TFKeras


class GCN(TFKeras):
    def __init__(self, in_channels, out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, bias=True):

        super().__init__()
        self.convs = []
        inc = in_channels
        for hid, act in zip(hids, acts):
            layer = GraphConv(inc, hid, bias=bias,
                              activation=activations.get(act))
            self.convs.append(layer)
            inc = hid

        layer = GraphConv(inc, out_channels, bias=bias)
        self.convs.append(layer)
        self.dropout = layers.Dropout(dropout)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])

    def call(self, inputs):
        h, g = inputs
        for layer in self.convs[:-1]:
            h = layer(g, h)
            h = self.dropout(h)
        h = self.convs[-1](g, h)

        return h
