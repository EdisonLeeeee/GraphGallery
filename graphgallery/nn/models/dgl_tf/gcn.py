import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from dgl.nn.tensorflow import GraphConv
from tensorflow.keras.activations import get


class GCN(Model):
    def __init__(self, in_channels, out_channels,
                 hiddens=[16],
                 activations=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, use_bias=True):

        super().__init__()
        self.convs = []
        inc = in_channels
        for hidden, activation in zip(hiddens, activations):
            layer = GraphConv(inc, hidden, bias=use_bias,
                              activation=get(activation))
            self.convs.append(layer)
            inc = hidden

        layer = GraphConv(inc, out_channels, bias=use_bias)
        self.convs.append(layer)
        self.dropout = layers.Dropout(dropout)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
        self.weight_decay = weight_decay
        self.metric = SparseCategoricalAccuracy()

    def call(self, inputs):
        h, g, idx = inputs
        for layer in self.convs[:-1]:
            h = layer(g, h)
            h = self.dropout(h)
        h = self.convs[-1](g, h)

        return tf.gather(h, idx)
