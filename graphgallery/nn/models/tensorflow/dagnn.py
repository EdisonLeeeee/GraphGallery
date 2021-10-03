from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import DAGNNConv
from graphgallery.nn.models.tf_engine import TFEngine
from graphgallery import floatx


class DAGNN(TFEngine):

    def __init__(self, in_features, out_features,
                 hids=[64], acts=['relu'],
                 dropout=0.5, weight_decay=5e-3,
                 lr=0.01, bias=False, K=10):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = Dense(hid, use_bias=bias, activation=act,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)
            h = Dropout(dropout)(h)

        h = Dense(out_features, use_bias=bias, activation=acts[-1],
                  kernel_regularizer=regularizers.l2(weight_decay))(h)
        h = Dropout(dropout)(h)

        h = DAGNNConv(K, use_bias=bias, activation='sigmoid',
                      kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr), metrics=['accuracy'])
