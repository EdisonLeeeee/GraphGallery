import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import MeanAggregator, GCNAggregator
from graphgallery.nn.models import SupervisedModel
from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery.utils.graph_utils import construct_neighbors
from graphgallery import astensor, asintarr, normalize_x


class GraphSAGE(SupervisedModel):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>


        Arguments:
        ----------
            adj: shape (N, N), Scipy sparse matrix if  `is_adj_sparse=True`, 
                Numpy array-like (or matrix) if `is_adj_sparse=False`.
                The input `symmetric` adjacency matrix, where `N` is the number 
                of nodes in graph.
            x: shape (N, F), Scipy sparse matrix if `is_x_sparse=True`, 
                Numpy array-like (or matrix) if `is_x_sparse=False`.
                The input node feature matrix, where `F` is the dimension of features.
            labels: Numpy array-like with shape (N,)
                The ground-truth labels for all nodes in graph.
            n_samples (List of positive integer, optional): 
                The number of sampled neighbors for each nodes in each layer. 
                (default :obj: `[10, 5]`, i.e., sample `10` first-order neighbors and 
                `5` sencond-order neighbors, and the radius for `GraphSAGE` is `2`)
            norm_x_type (String, optional): 
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :str: `l1`)
            device (String, optional): 
                The device where the model is running on. You can specified `CPU` or `GPU` 
                for the model. (default: :str: `CPU:0`, i.e., running on the 0-th `CPU`)
            seed (Positive integer, optional): 
                Used in combination with `tf.random.set_seed` & `np.random.seed` & `random.seed`  
                to create a reproducible sequence of tensors across multiple calls. 
                (default :obj: `None`, i.e., using random seed)
            name (String, optional): 
                Specified name for the model. (default: :str: `class.__name__`)


    """

    def __init__(self, adj, x, labels, n_samples=[15, 5], norm_x_type='l1',
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

        self.n_samples = n_samples
        self.norm_x_type = norm_x_type
        self.preprocess(adj, x)

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        # check the input adj and x, and convert them into proper data types
        adj, x = self._check_inputs(adj, x)

        if self.norm_x_type:
            x = normalize_x(x, norm=self.norm_x_type)

        # Dense matrix, shape [n_nodes, max_degree]
        neighbors = construct_neighbors(adj, max_degree=max(self.n_samples))
        # pad with a dummy zero vector
        x = np.vstack([x, np.zeros(self.n_features, dtype=self.floatx)])

        with tf.device(self.device):
            self.x_norm, self.neighbors = astensor(x), neighbors

    def build(self, hiddens=[64], activations=['relu'], dropout=0.5, lr=0.01, l2_norm=5e-4,
              output_normalize=False, aggrator='mean'):

        assert len(hiddens) == len(self.n_samples)-1, "The number of hidden units and " \
            "samples per layer should be the same"
        assert len(hiddens) == len(activations), "The number of hidden units and " \
            "activation functions should be the same."

        with tf.device(self.device):

            if aggrator == 'mean':
                Agg = MeanAggregator
            elif aggrator == 'gcn':
                Agg = GCNAggregator
            else:
                raise ValueError(f'Invalid value of `aggrator`, allowed values (`mean`, `gcn`), but got `{aggrator}`.')

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            nodes = Input(batch_shape=[None], dtype=self.intx, name='nodes')
            neighbors = [Input(batch_shape=[None], dtype=self.intx, name=f'neighbors_{hop}')
                         for hop, n_sample in enumerate(self.n_samples)]

            aggrators = []
            for i, (hid, activation) in enumerate(zip(hiddens, activations)):
                # you can use `GCNAggregator` instead
                aggrators.append(Agg(hid, concat=True, activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm)))

            aggrators.append(Agg(self.n_classes))

            h = [tf.nn.embedding_lookup(x, node) for node in [nodes, *neighbors]]
            for agg_i, aggrator in enumerate(aggrators):
                feature_shape = h[0].shape[-1]
                for hop in range(len(self.n_samples)-agg_i):
                    neighbor_shape = [-1, self.n_samples[hop], feature_shape]
                    h[hop] = Dropout(rate=dropout)(aggrator([h[hop], tf.reshape(h[hop+1], neighbor_shape)]))
                h.pop()

            h = h[0]
            if output_normalize:
                h = tf.nn.l2_normalize(h, axis=1)
            output = Softmax()(h)

            model = Model(inputs=[x, nodes, *neighbors], outputs=output)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.set_model(model)

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = SAGEMiniBatchSequence([self.x_norm, index], labels, neighbors=self.neighbors, n_samples=self.n_samples)
        return sequence

    def predict(self, index):
        super().predict(index)
        logit = []
        index = asintarr(index)
        with tf.device(self.device):
            data = SAGEMiniBatchSequence([self.x_norm, index], neighbors=self.neighbors, n_samples=self.n_samples)
            for inputs, labels in data:
                output = self.model.predict_on_batch(inputs)
                if tf.is_tensor(output):
                    output = output.numpy()

                logit.append(output)
        logit = np.concatenate(logit, axis=0)
        return logit
