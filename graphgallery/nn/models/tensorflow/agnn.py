from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery.nn.layers.tensorflow import AGNNConv
from graphgallery import floatx
from graphgallery.nn.models.tf_engine import TFEngine


class AGNN(TFEngine):

    def __init__(self, in_features, out_features,
                 hids=[16],
                 acts=['relu'],
                 num_attn=3,
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, bias=False):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = Dense(hid, use_bias=bias, activation=act,
                      kernel_regularizer=regularizers.l2(weight_decay))(h)
            h = Dropout(rate=dropout)(h)
        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0
        h = AGNNConv(trainable=False, regularizer=regularizers.l2(weight_decay))([h, adj])
        for _ in range(1, num_attn):
            h = AGNNConv(regularizer=regularizers.l2(weight_decay))([h, adj])

        h = Dense(out_features, use_bias=bias)(h)
        h = Dropout(rate=dropout)(h)

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr), metrics=['accuracy'])
