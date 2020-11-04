from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import WaveletConvolution, Gather
from graphgallery import floatx, intx


class GWNN(Model):

    def __init__(self, in_channels, out_channels, n_nodes,
                 hiddens=[16], activations=['relu'],
                 dropout=0.5, l2_norm=5e-4, lr=0.01,
                 use_bias=False):

        _floatx = floatx()
        x = Input(batch_shape=[None, in_channels],
                  dtype=_floatx, name='attr_matrix')
        wavelet = Input(batch_shape=[n_nodes, n_nodes],
                        dtype=_floatx, sparse=True,
                        name='wavelet_matrix')
        inverse_wavelet = Input(batch_shape=[n_nodes, n_nodes],
                                dtype=_floatx, sparse=True,
                                name='inverse_wavelet_matrix')
        index = Input(batch_shape=[None],
                      dtype=intx(), name='node_index')

        h = x
        for hidden, activation in zip(hiddens, activations):
            h = WaveletConvolution(hidden, activation=activation, use_bias=use_bias,
                                   kernel_regularizer=regularizers.l2(l2_norm))([h, wavelet, inverse_wavelet])
            h = Dropout(rate=dropout)(h)

        h = WaveletConvolution(out_channels, use_bias=use_bias)(
            [h, wavelet, inverse_wavelet])
        h = Gather()([h, index])

        super().__init__(inputs=[x, wavelet, inverse_wavelet, index], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
