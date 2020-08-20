import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import MeanAggregator, GCNAggregator
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery.utils.graph import construct_neighbors
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import astensors, asintarr, normalize_x, Bunch


class GraphSAGE(SemiSupervisedModel):
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
            norm_x (String, optional): 
                How to normalize the node feature matrix. See `graphgallery.normalize_x`
                (default :obj: `None`)
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

    def __init__(self, adj, x, labels, n_samples=[15, 5], norm_x=None,
                 device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels, device=device, seed=seed, name=name, **kwargs)

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
        x = np.vstack([x, np.zeros(self.n_features, dtype=self.floatx)])

        with tf.device(self.device):
            self.x_norm, self.neighbors = astensors(x), neighbors

    def build(self, hiddens=[32], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4], lr=0.01,
              use_bias=True, output_normalize=False, aggrator='mean'):

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
            for i, (hid, activation, l2_norm) in enumerate(zip(hiddens, activations, l2_norms)):
                # you can use `GCNAggregator` instead
                aggrators.append(Agg(hid, concat=True, activation=activation, use_bias=use_bias,
                                     kernel_regularizer=regularizers.l2(l2_norm)))

            aggrators.append(Agg(self.n_classes, use_bias=use_bias))

            h = [tf.nn.embedding_lookup(x, node) for node in [nodes, *neighbors]]
            for agg_i, aggrator in enumerate(aggrators):
                feature_shape = h[0].shape[-1]
                for hop in range(len(self.n_samples)-agg_i):
                    neighbor_shape = [-1, self.n_samples[hop], feature_shape]
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
