from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphConvolution
from graphgallery.nn.models import TFKeras
from graphgallery import floatx


class FastGCN(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=False):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = Dense(hid, use_bias=bias, activation=act,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)
            h = Dropout(rate=dropout)(h)

        h = GraphConvolution(out_features,
                             use_bias=bias)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
