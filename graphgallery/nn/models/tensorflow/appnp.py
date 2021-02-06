from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery.nn.layers.tensorflow import APPNPropagation, PPNPropagation
from graphgallery import floatx
from graphgallery.nn.models import TFKeras


class APPNP(TFKeras):

    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha=0.1,
                 K=10,
                 ppr_dropout=0.,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=True,
                 approximated=True):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=approximated, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = Dense(hid, use_bias=use_bias,
                      activation=act,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)

            h = Dropout(rate=dropout)(h)

        h = Dense(out_channels, use_bias=use_bias,
                  kernel_regularizer=regularizers.l2(weight_decay))(h)
        if approximated:
            h = APPNPropagation(alpha=alpha, K=K, dropout=ppr_dropout)([h, adj])
        else:
            h = PPNPropagation(dropout=ppr_dropout)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
