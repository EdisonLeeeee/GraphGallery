import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import MedianAggregator, MedianGCNAggregator
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery.utils.graph import construct_neighbors
from graphgallery.utils.shape import EqualVarLength
from graphgallery import astensors, asintarr, normalize_x, Bunch


class MedianSAGE(SemiSupervisedModel):

    def __init__(self, graph, n_samples=[15, 3], norm_x=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """
        Parameters:
        ----------
            graph: graphgallery.data.Graph
                A sparse, attributed, labeled graph.
            n_samples: List of positive integer. optional 
                The number of sampled neighbors for each nodes in each layer. 
                (default :obj: `[10, 5]`, i.e., sample `10` first-order neighbors and 
                `5` sencond-order neighbors, and the radius for `GraphSAGE` is `2`)
            norm_x: string. optional 
                How to normalize the node attribute matrix. See `graphgallery.normalize_x`
                (default :obj: `None`)
            device: string. optional 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
            seed: interger scalar. optional 
                Used in combination with `tf.random.set_seed` & `np.random.seed` 
                & `random.seed` to create a reproducible sequence of tensors across 
                multiple calls. (default :obj: `None`, i.e., using random seed)
            name: string. optional
                Specified name for the model. (default: :str: `class.__name__`)
            kwargs: other customed keyword Parameters.


        """
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

        self.n_samples = n_samples
        self.norm_x = norm_x
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        # Dense matrix, shape [n_nodes, max_degree]
        neighbors = construct_neighbors(adj, max_degree=max(self.n_samples))
        # pad with a dummy zero vector
        x = np.vstack([x, np.zeros(self.n_attrs, dtype=self.floatx)])

        with tf.device(self.device):
            x = astensors(x)
            self.x_norm, self.neighbors = x, neighbors

    def build(self, hiddens=[32], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4], lr=0.01,
              use_bias=True, output_normalize=False, aggrator='median'):

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms,
                                                                       max_length=len(self.n_samples)-1)
        paras.update(Bunch(hiddens=hiddens, activations=activations,
                           dropouts=dropouts, l2_norms=l2_norms, n_samples=self.n_samples))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)
        ######################################

        with tf.device(self.device):

            if aggrator == 'median':
                Agg = MedianAggregator
            elif aggrator == 'gcn':
                Agg = MedianGCNAggregator
            else:
                raise ValueError(f'Invalid value of `aggrator`, allowed values (`median`, `gcn`), but got `{aggrator}`')

            x = Input(batch_shape=[None, self.n_attrs], dtype=self.floatx, name='attr_matrix')
            nodes = Input(batch_shape=[None], dtype=self.intx, name='nodes')
            neighbors = [Input(batch_shape=[None], dtype=self.intx, name=f'neighbors_{hop}')
                         for hop, n_sample in enumerate(self.n_samples)]

            aggrators = []
            for i, (hid, activation, l2_norm) in enumerate(zip(hiddens, activations, l2_norms)):
                # you can use `GCNAggregator` instead
                aggrators.append(Agg(hid, concat=True, activation=activation, use_bias=use_bias,
                                     kernel_regularizer=regularizers.l2(l2_norm)))

            aggrators.append(Agg(self.n_classes, use_bias=use_bias))

            h = [tf.nn.embedding_lookup(x, node) for node in [nodes, *neighbors]]
            for agg_i, aggrator in enumerate(aggrators):
                attribute_shape = h[0].shape[-1]
                for hop in range(len(self.n_samples)-agg_i):
                    neighbor_shape = [-1, self.n_samples[hop], attribute_shape]
                    h[hop] = aggrator([h[hop], tf.reshape(h[hop+1], neighbor_shape)])
                    if hop != len(self.n_samples)-1:
                        h[hop] = Dropout(rate=dropouts[hop])(h[hop])
                h.pop()

            h = h[0]
            if output_normalize:
                h = tf.nn.l2_normalize(h, axis=1)

            model = Model(inputs=[x, nodes, *neighbors], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.model = model

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = SAGEMiniBatchSequence([self.x, index], labels, self.neighbors, n_samples=self.n_samples)
        return sequence

    def predict(self, index):
        super().predict(index)
        logit = []
        index = asintarr(index)
        with tf.device(self.device):
            data = SAGEMiniBatchSequence([self.x, index], neighbors=self.neighbors, n_samples=self.n_samples)
            for inputs, labels in data:
                output = self.model.predict_on_batch(inputs)
                if tf.is_tensor(output):
                    output = output.numpy()

                logit.append(output)
        logit = np.concatenate(logit, axis=0)
        return logit
