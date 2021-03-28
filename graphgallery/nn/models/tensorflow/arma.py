from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphgallery.nn.layers.tensorflow import ARMAConv
from graphgallery import floatx
from graphgallery.nn.models import TFKeras


class ARMA(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[16],
                 acts=['elu'],
                 dropout=0.5,
                 order=2,
                 iterations=1,
                 weight_decay=5e-5,
                 share_weights=True,
                 lr=0.01, bias=True):

        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=True, name='adj_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = ARMAConv(hid, use_bias=bias,
                         activation=act,
                         order=order,
                         iterations=iterations,
                         gcn_activation="elu",
                         share_weights=share_weights,
                         kernel_regularizer=regularizers.l2(weight_decay))([h, adj])
            h = Dropout(rate=dropout)(h)

        h = ARMAConv(out_features,
                     use_bias=bias,
                     order=1,
                     iterations=1,
                     gcn_activation=None,
                     share_weights=share_weights)([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
