from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import WaveletConv
from graphgallery import floatx
from graphgallery.nn.models.tf_keras import TFKeras


class GWNN(TFKeras):

    def __init__(self, in_features, out_features, num_nodes,
                 hids=[16], acts=['relu'],
                 dropout=0.5, weight_decay=5e-4, lr=0.01,
                 bias=False):

        _floatx = floatx()
        x = Input(batch_shape=[None, in_features],
                  dtype=_floatx, name='node_attr')
        wavelet = Input(batch_shape=[num_nodes, num_nodes],
                        dtype=_floatx, sparse=True,
                        name='wavelet_matrix')
        inverse_wavelet = Input(batch_shape=[num_nodes, num_nodes],
                                dtype=_floatx, sparse=True,
                                name='inverse_wavelet_matrix')

        h = x
        for hid, act in zip(hids, acts):
            h = WaveletConv(hid, activation=act, use_bias=bias,
                            kernel_regularizer=regularizers.l2(weight_decay))([h, wavelet, inverse_wavelet])
            h = Dropout(rate=dropout)(h)

        h = WaveletConv(out_features, use_bias=bias)(
            [h, wavelet, inverse_wavelet])

        super().__init__(inputs=[x, wavelet, inverse_wavelet], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
