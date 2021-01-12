from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import DenseConvolution
from graphgallery.nn.models import TFKeras
from graphgallery import floatx


class DenseGCN(TFKeras):

    def __init__(self, in_channels, out_channels,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, use_bias=False):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=False, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = DenseConvolution(hid, use_bias=use_bias,
                                 activation=act,
                                 kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

            h = Dropout(rate=dropout)(h)

        h = DenseConvolution(out_channels, use_bias=use_bias)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
