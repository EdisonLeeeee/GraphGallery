import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GaussionConvF, GaussionConvD, Sample
from graphgallery import floatx
from graphgallery.nn.models.tf_keras import TFKeras


class RobustGCN(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01, kl=5e-4, gamma=1.,
                 bias=False):

        _floatx = floatx()
        x = Input(batch_shape=[None, in_features],
                  dtype=_floatx, name='node_attr')
        adj = [Input(batch_shape=[None, None], dtype=_floatx,
                     sparse=True, name='adj_matrix_1'),
               Input(batch_shape=[None, None], dtype=_floatx, sparse=True,
                     name='adj_matrix_2')]

        h = x
        if hids:
            mean, var = GaussionConvF(hids[0], gamma=gamma,
                                      use_bias=bias,
                                      activation=acts[0],
                                      kernel_regularizer=regularizers.l2(weight_decay))([h, *adj])
            if kl:
                KL_divergence = 0.5 * \
                    tf.reduce_mean(tf.math.square(mean) + var -
                                   tf.math.log(1e-8 + var) - 1, axis=1)
                KL_divergence = tf.reduce_sum(KL_divergence)

                # KL loss
                kl_loss = kl * KL_divergence

        # additional layers (usually unnecessay)
        for hid, act in zip(hids[1:], acts[1:]):

            mean, var = GaussionConvD(
                hid, gamma=gamma, use_bias=bias, activation=act)([mean, var, *adj])
            mean = Dropout(rate=dropout)(mean)
            var = Dropout(rate=dropout)(var)

        mean, var = GaussionConvD(
            out_features, gamma=gamma, use_bias=bias)([mean, var, *adj])

        h = Sample()([mean, var])

        super().__init__(inputs=[x, *adj], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])

        if hids and kl:
            self.add_loss(kl_loss)
