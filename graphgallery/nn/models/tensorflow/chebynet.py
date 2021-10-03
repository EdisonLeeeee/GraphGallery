from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import ChebConv
from graphgallery.nn.models.tf_engine import TFEngine
from graphgallery import floatx


class ChebyNet(TFEngine):

    def __init__(self, in_features, out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, K=2, bias=False):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = [Input(batch_shape=[None, None],
                     dtype=floatx(), sparse=True,
                     name=f'adj_matrix_{i}') for i in range(K + 1)]

        h = x
        for hid, act in zip(hids, acts):
            h = ChebConv(hid, K=K, use_bias=bias,
                         activation=act,
                         kernel_regularizer=regularizers.l2(weight_decay))([h, adj])
            h = Dropout(rate=dropout)(h)

        h = ChebConv(out_features,
                     K=K, use_bias=bias)([h, adj])

        super().__init__(inputs=[x, *adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr), metrics=['accuracy'])
