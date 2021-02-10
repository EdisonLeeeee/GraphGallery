from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import Top_k_features, LGConvolution, DenseConvolution
from graphgallery.nn.models import TFKeras
from graphgallery import floatx


class LGCN(TFKeras):

    def __init__(self, in_channels, out_channels,
                 hids=[32], num_filters=[8, 8],
                 acts=[None, None],
                 dropout=0.8,
                 weight_decay=5e-4, lr=0.1, bias=False, K=8):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=False, name='adj_matrix')

        h = x
        for idx, hid in enumerate(hids):
            h = Dropout(rate=dropout)(h)
            h = DenseConvolution(hid,
                                 use_bias=bias,
                                 activation=acts[idx],
                                 kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

        for idx, num_filter in enumerate(num_filters):
            top_k_h = Top_k_features(K=K)([h, adj])
            cur_h = LGConvolution(num_filter,
                                  kernel_size=K,
                                  use_bias=bias,
                                  dropout=dropout,
                                  activation=acts[idx],
                                  kernel_regularizer=regularizers.l2(weight_decay))(top_k_h)
            cur_h = BatchNormalization()(cur_h)
            h = Concatenate()([h, cur_h])

        h = Dropout(rate=dropout)(h)
        h = DenseConvolution(out_channels,
                             use_bias=bias,
                             activation=acts[-1],
                             kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

        super().__init__(inputs=[x, adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Nadam(lr=lr), metrics=['accuracy'])
