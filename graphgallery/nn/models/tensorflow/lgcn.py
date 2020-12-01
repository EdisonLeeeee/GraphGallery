from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import Top_k_features, LGConvolution, DenseConvolution, Mask
from graphgallery import floatx, intx
from graphgallery.nn.models import TFKeras


class LGCN(TFKeras):

    def __init__(self, in_channels, out_channels,
                 hiddens=[32], n_filters=[8, 8],
                 activations=[None, None],
                 dropout=0.8,
                 weight_decay=5e-4, lr=0.1, use_bias=False, K=8):

        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=floatx(),
                    sparse=False, name='adj_matrix')
        mask = Input(batch_shape=[None], dtype='bool', name='node_mask')

        h = x
        for idx, hidden in enumerate(hiddens):
            h = Dropout(rate=dropout)(h)
            h = DenseConvolution(hidden,
                                 use_bias=use_bias,
                                 activation=activations[idx],
                                 kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

        for idx, n_filter in enumerate(n_filters):
            top_k_h = Top_k_features(K=K)([h, adj])
            cur_h = LGConvolution(n_filter,
                                  kernel_size=K,
                                  use_bias=use_bias,
                                  dropout=dropout,
                                  activation=activations[idx],
                                  kernel_regularizer=regularizers.l2(weight_decay))(top_k_h)
            cur_h = BatchNormalization()(cur_h)
            h = Concatenate()([h, cur_h])

        h = Dropout(rate=dropout)(h)
        h = DenseConvolution(out_channels,
                             use_bias=use_bias,
                             activation=activations[-1],
                             kernel_regularizer=regularizers.l2(weight_decay))([h, adj])

        h = Mask()([h, mask])

        super().__init__(inputs=[x, adj, mask], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Nadam(lr=lr), metrics=['accuracy'])
