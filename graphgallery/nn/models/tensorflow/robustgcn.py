import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GaussionConvolution_F, GaussionConvolution_D, Sample, Gather
from graphgallery import floatx, intx


class RobustGCN(Model):

    def __init__(self, in_channels, out_channels,
                 hiddens=[64],
                 activations=['relu'],
                 dropout=0.5,
                 l2_norm=5e-4,
                 lr=0.01, kl=5e-4, gamma=1.,
                 use_bias=False):

        _floatx = floatx()
        x = Input(batch_shape=[None, in_channels],
                  dtype=_floatx, name='attr_matrix')
        adj = [Input(batch_shape=[None, None], dtype=_floatx,
                     sparse=True, name='adj_matrix_1'),
               Input(batch_shape=[None, None], dtype=_floatx, sparse=True,
                     name='adj_matrix_2')]
        index = Input(batch_shape=[None], dtype=intx(), name='node_index')

        h = x
        if hiddens:
            mean, var = GaussionConvolution_F(hiddens[0], gamma=gamma,
                                              use_bias=use_bias,
                                              activation=activations[0],
                                              kernel_regularizer=regularizers.l2(l2_norm))([h, *adj])
            if kl:
                KL_divergence = 0.5 * \
                    tf.reduce_mean(tf.math.square(mean) + var -
                                   tf.math.log(1e-8 + var) - 1, axis=1)
                KL_divergence = tf.reduce_sum(KL_divergence)

                # KL loss
                kl_loss = kl * KL_divergence

        # additional layers (usually unnecessay)
        for hidden, activation in zip(hiddens[1:], activations[1:]):

            mean, var = GaussionConvolution_D(
                hidden, gamma=gamma, use_bias=use_bias, activation=activation)([mean, var, *adj])
            mean = Dropout(rate=dropout)(mean)
            var = Dropout(rate=dropout)(var)

        mean, var = GaussionConvolution_D(
            out_channels, gamma=gamma, use_bias=use_bias)([mean, var, *adj])

        h = Sample()([mean, var])
        h = Gather()([h, index])

        super().__init__(inputs=[x, *adj, index], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])

        if hiddens and kl:
            self.add_loss(kl_loss)
