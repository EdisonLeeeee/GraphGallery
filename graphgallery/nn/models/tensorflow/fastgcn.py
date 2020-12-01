from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphConvolution, Gather
from graphgallery import floatx, intx
from graphgallery.nn.models import TFKeras


class FastGCN(TFKeras):

    def __init__(self, in_channels, out_channels,
                 hiddens=[32], activations=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, use_bias=False):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hidden, activation in zip(hiddens, activations):
            h = Dense(hidden, use_bias=use_bias, activation=activation,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)
            h = Dropout(rate=dropout)(h)

        h = GraphConvolution(out_channels,
                             use_bias=use_bias)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
