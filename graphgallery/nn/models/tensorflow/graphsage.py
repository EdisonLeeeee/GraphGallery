import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import MeanAggregator, GCNAggregator, MedianAggregator, MedianGCNAggregator
from graphgallery import floatx, intx
from graphgallery.nn.models import TFKeras


_AGG = {'mean': MeanAggregator,
        'gcn': GCNAggregator,
        'median': MedianAggregator,
        'mediangcn': MedianGCNAggregator
        }


class GraphSAGE(TFKeras):

    def __init__(self, in_features, out_features,
                 hids=[32], acts=['relu'], dropout=0.5,
                 weight_decay=5e-4, lr=0.01, bias=True,
                 aggregator='mean', output_normalize=False, num_samples=[15, 5]):

        Agg = _AGG.get(aggregator, None)
        if not Agg:
            raise ValueError(
                f"Invalid value of 'aggregator', allowed values {tuple(_AGG.keys())}, but got '{aggregator}'.")

        _intx = intx()
        x = Input(batch_shape=[None, in_features],
                  dtype=floatx(), name='node_attr')
        nodes = Input(batch_shape=[None], dtype=_intx, name='nodes')
        neighbors = [Input(batch_shape=[None], dtype=_intx, name=f'neighbors_{hop}')
                     for hop, num_sample in enumerate(num_samples)]

        aggregators = []
        for hid, act in zip(hids, acts):
            # you can use `GCNAggregator` instead
            aggregators.append(Agg(hid, concat=True, activation=act,
                                   use_bias=bias,
                                   kernel_regularizer=regularizers.l2(weight_decay)))

        aggregators.append(Agg(out_features, use_bias=bias))

        h = [tf.nn.embedding_lookup(x, node)
             for node in [nodes, *neighbors]]
        for agg_i, aggregator in enumerate(aggregators):
            attribute_shape = h[0].shape[-1]
            for hop in range(len(num_samples) - agg_i):
                neighbor_shape = [-1, num_samples[hop], attribute_shape]
                h[hop] = aggregator(
                    [h[hop], tf.reshape(h[hop + 1], neighbor_shape)])
                if hop != len(num_samples) - 1:
                    h[hop] = Dropout(rate=dropout)(h[hop])
            h.pop()

        h = h[0]
        if output_normalize:
            h = tf.nn.l2_normalize(h, axis=1)

        super().__init__(inputs=[x, nodes, *neighbors], outputs=h)
        self.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=Adam(lr=lr), metrics=['accuracy'])
