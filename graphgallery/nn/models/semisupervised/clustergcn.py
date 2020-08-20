import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import GraphConvolution, SparseConversion
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import ClusterMiniBatchSequence
from graphgallery.utils.graph import partition_graph
from graphgallery.utils.shape import set_equal_in_length
from graphgallery import Bunch, sample_mask, normalize_x, normalize_adj, astensors, asintarr, sparse_adj_to_edges


class ClusterGCN(SemiSupervisedModel):
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
            norm_adj (Float scalar, optional):
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`,
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
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

    def __init__(self, adj, x, labels, graph=None, n_clusters=None,
                 norm_adj=-0.5, norm_x=None, device='CPU:0', seed=None, name=None, **kwargs):

        super().__init__(adj, x, labels=labels, device=device, seed=seed, name=name, **kwargs)

        if not n_clusters:
            n_clusters = self.n_classes

        self.n_clusters = n_clusters
        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x, graph)

    def preprocess(self, adj, x, graph=None):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        if graph is None:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)

        batch_adj, batch_x, self.cluster_member = partition_graph(adj, x, graph,
                                                                  n_clusters=self.n_clusters)

        if self.norm_adj:
            batch_adj = normalize_adj(batch_adj, self.norm_adj)

        batch_adj = [sparse_adj_to_edges(b_adj) for b_adj in batch_adj]
        batch_edge_index, batch_edge_weight = tuple(zip(*batch_adj))

        with tf.device(self.device):
            (self.batch_edge_index,
             self.batch_edge_weight,
             self.batch_x) = astensors([batch_edge_index, batch_edge_weight, batch_x])

    def build(self, hiddens=[32], activations=['relu'], dropouts=[0.5], l2_norms=[1e-5], lr=0.01,
              use_bias=False):

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)
        ######################################

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_features], dtype=self.floatx, name='features')
            edge_index = Input(batch_shape=[None, 2], dtype=tf.int64, name='edge_index')
            edge_weight = Input(batch_shape=[None], dtype=self.floatx, name='edge_weight')
            mask = Input(batch_shape=[None],  dtype=tf.bool, name='mask')

            adj = SparseConversion()([edge_index, edge_weight])

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = Dropout(rate=dropout)(h)
                h = GraphConvolution(hid, use_bias=use_bias, activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

            h = Dropout(rate=dropout)(h)
            h = GraphConvolution(self.n_classes, use_bias=use_bias)([h, adj])
            h = tf.boolean_mask(h, mask)

            model = Model(inputs=[x, edge_index, edge_weight, mask], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.model = model

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)
        mask = sample_mask(index, self.n_nodes)

        orders_dict = {idx: order for order, idx in enumerate(index)}
        batch_mask, orders = [], []
        batch_x, batch_edge_index, batch_edge_weight = [], [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            batch_nodes = np.asarray(nodes)[mini_mask]
            if batch_nodes.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_edge_index.append(self.batch_edge_index[cluster])
            batch_edge_weight.append(self.batch_edge_weight[cluster])
            batch_mask.append(mini_mask)
            orders.append([orders_dict[n] for n in batch_nodes])

        batch_data = tuple(zip(batch_x, batch_edge_index, batch_edge_weight, batch_mask))

        logit = np.zeros((index.size, self.n_classes), dtype=self.floatx)
        with tf.device(self.device):
            batch_data = astensors(batch_data)
            for order, inputs in zip(orders, batch_data):
                output = self.model.predict_on_batch(inputs)
                logit[order] = output

        return logit

    def train_sequence(self, index):
        index = asintarr(index)
        mask = sample_mask(index, self.n_nodes)
        labels = self.labels

        batch_mask, batch_labels = [], []
        batch_x, batch_edge_index, batch_edge_weight = [], [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            mini_labels = labels[nodes][mini_mask]
            if mini_labels.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_edge_index.append(self.batch_edge_index[cluster])
            batch_edge_weight.append(self.batch_edge_weight[cluster])
            batch_mask.append(mini_mask)
            batch_labels.append(mini_labels)

        batch_data = tuple(zip(batch_x, batch_edge_index, batch_edge_weight, batch_mask))

        with tf.device(self.device):
            sequence = ClusterMiniBatchSequence(batch_data, batch_labels)
        return sequence
