import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvolution
from graphgallery.nn.models import SupervisedModel
from graphgallery.sequence import ClusterMiniBatchSequence
from graphgallery.utils.graph_utils import partition_graph
from graphgallery import Bunch, sample_mask, normalize_x, normalize_adj, astensor, asintarr


class ClusterGCN(SupervisedModel):
    """
        Implementation of Cluster Graph Convolutional Networks (ClusterGCN). 

        `Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks 
        <https://arxiv.org/abs/1905.07953>`
        Tensorflow 1.x implementation: <https://github.com/google-research/google-research/tree/master/cluster_gcn>
        Pytorch implementation: <https://github.com/benedekrozemberczki/ClusterGCN>

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
            graph (`nx.DiGraph`, optional): 
                The networkx graph which converted by `adj`, if if not specified (`None`), 
                the graph will be converted by `adj` automatically, but it will comsum lots 
                of time. (default :obj: `None`)
            n_clusters (Potitive integer): 
                The number of clusters that the graph being seperated, if not specified (`None`), 
                it will be set to the number of classes automatically. (default :obj: `None`).
            norm_adj_rate (Float scalar, optional): 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
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

    def __init__(self, adj, x, labels, graph=None, n_clusters=None,
                 norm_adj_rate=-0.5, norm_x_type='l1', device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels=labels, device=device, seed=seed, name=name, **kwargs)

        if n_clusters is None:
            n_clusters = self.n_classes

        self.n_clusters = n_clusters
        self.norm_adj_rate = norm_adj_rate
        self.norm_x_type = norm_x_type
        self.preprocess(adj, x, graph)

    def preprocess(self, adj, x, graph=None):
        super().preprocess(adj, x)
        # check the input adj and x, and convert them into proper data types
        adj, x = self._check_inputs(adj, x)

        if self.norm_x_type:
            x = normalize_x(x, norm=self.norm_x_type)

        if not graph:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)

        (self.batch_adj, self.batch_x, self.batch_labels,
         self.cluster_member, self.mapper) = partition_graph(adj, x, self.labels, graph,
                                                             n_clusters=self.n_clusters)

        if self.norm_adj_rate:
            self.batch_adj = normalize_adj(self.batch_adj, self.norm_adj_rate)

        with tf.device(self.device):
            self.batch_adj, self.batch_x = astensor([self.batch_adj, self.batch_x])

    def build(self, hiddens=[32], activations=['relu'], dropout=0.5, lr=0.01, l2_norm=1e-5):

        assert len(hiddens) == len(activations), "The number of hidden units and " \
            "activation functions should be the same."

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            mask = Input(batch_shape=[None],  dtype=tf.bool, name='mask')

            h = Dropout(rate=dropout)(x)

            for hid, activation in zip(hiddens, activations):
                h = GraphConvolution(hid, activation=activation, kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = GraphConvolution(self.n_classes)([h, adj])
            h = tf.boolean_mask(h, mask)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, mask], outputs=output)

            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr),
                          metrics=['accuracy'], experimental_run_tf_function=False)

            self.set_model(model)

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)
        mask = sample_mask(index, self.n_nodes)

        orders_dict = {idx: order for order, idx in enumerate(index)}
        batch_mask, orders = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            batch_nodes = np.asarray(nodes)[mini_mask]
            if batch_nodes.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_adj.append(self.batch_adj[cluster])
            batch_mask.append(mini_mask)
            orders.append([orders_dict[n] for n in batch_nodes])

        batch_data = tuple(zip(batch_x, batch_adj, batch_mask))

        logit = np.zeros((index.size, self.n_classes), dtype=self.floatx)
        with tf.device(self.device):
            batch_data = astensor(batch_data)
            for order, inputs in zip(orders, batch_data):
                output = self.model.predict_on_batch(inputs)
                logit[order] = output

        return logit

    def train_sequence(self, index):
        index = asintarr(index)
        mask = sample_mask(index, self.n_nodes)
        labels = self.labels

        batch_mask, batch_labels = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            mini_labels = labels[nodes][mini_mask]
            if mini_labels.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_adj.append(self.batch_adj[cluster])
            batch_mask.append(mini_mask)
            batch_labels.append(mini_labels)

        batch_data = tuple(zip(batch_x, batch_adj, batch_mask))
        with tf.device(self.device):
            sequence = ClusterMiniBatchSequence(batch_data, batch_labels)
        return sequence
