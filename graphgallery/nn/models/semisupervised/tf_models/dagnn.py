from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tf_layers import PropConvolution, Gather
from graphgallery import floatx, intx


class DAGNN(Model):

    def __init__(self, in_channels, out_channels,
                 hiddens=[64], activations=['relu'], 
                 dropout=0.5, l2_norm=5e-3,
                 lr=0.01, use_bias=False, K=10):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='attr_matrix')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')
        index = Input(batch_shape=[None], dtype=intx(), name='node_index')

        h = x
        for hidden, activation in zip(hiddens, activations):
            h = Dense(hidden, use_bias=use_bias, activation=activation,
                        kernel_regularizer=regularizers.l2(l2_norm))(h)
            h = Dropout(dropout)(h)

        h = Dense(out_channels, use_bias=use_bias, activation=activations[-1],
                    kernel_regularizer=regularizers.l2(l2_norm))(h)
        h = Dropout(dropout)(h)

        h = PropConvolution(K, use_bias=use_bias, activation='sigmoid',
                            kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
        h = Gather()([h, index])

        super().__init__(inputs=[x, adj, index], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])

