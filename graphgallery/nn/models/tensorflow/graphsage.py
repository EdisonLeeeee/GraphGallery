import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import MeanAggregator, GCNAggregator, MedianAggregator, MedianGCNAggregator
from graphgallery import floatx, intx


_AGG = {'mean': MeanAggregator,
        'gcn': GCNAggregator,
        'median': MedianAggregator,
        'mediangcn': MedianGCNAggregator
        }


class GraphSAGE(Model):

    def __init__(self, in_channels, out_channels,
                 hiddens=[32], activations=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, use_bias=True,
                 aggregator='mean', output_normalize=False, n_samples=[15, 5]):

        Agg = _AGG.get(aggregator, None)
        if not Agg:
            raise ValueError(
                f"Invalid value of 'aggregator', allowed values {tuple(_AGG.keys())}, but got '{aggregator}'.")

        _intx = intx()
        x = Input(batch_shape=[None, in_channels],
                  dtype=floatx(), name='node_attr')
        nodes = Input(batch_shape=[None], dtype=_intx, name='nodes')
        neighbors = [Input(batch_shape=[None], dtype=_intx, name=f'neighbors_{hop}')
                     for hop, n_sample in enumerate(n_samples)]

        aggregators = []
        for hidden, activation in zip(hiddens, activations):
            # you can use `GCNAggregator` instead
            aggregators.append(Agg(hidden, concat=True, activation=activation,
                                   use_bias=use_bias,
                                   kernel_regularizer=regularizers.l2(weight_decay)))

        aggregators.append(Agg(out_channels, use_bias=use_bias))

        h = [tf.nn.embedding_lookup(x, node)
             for node in [nodes, *neighbors]]
        for agg_i, aggregator in enumerate(aggregators):
            attribute_shape = h[0].shape[-1]
            for hop in range(len(n_samples) - agg_i):
                neighbor_shape = [-1, n_samples[hop], attribute_shape]
                h[hop] = aggregator(
                    [h[hop], tf.reshape(h[hop + 1], neighbor_shape)])
                if hop != len(n_samples) - 1:
                    h[hop] = Dropout(rate=dropout)(h[hop])
            h.pop()

        h = h[0]
        if output_normalize:
            h = tf.nn.l2_normalize(h, axis=1)

        super().__init__(inputs=[x, nodes, *neighbors], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
