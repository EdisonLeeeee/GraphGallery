import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tf_layers import GraphConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import MiniBatchSequence
from graphgallery.utils.decorators import EqualVarLength
from graphgallery import transformers as T


class ClusterGCN(SemiSupervisedModel):
    """
        Implementation of Cluster Graph Convolutional Networks (ClusterGCN).

        `Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
        <https://arxiv.org/abs/1905.07953>`
        Tensorflow 1.x implementation: 
        <https://github.com/google-research/google-research/tree/master/cluster_gcn>
        Pytorch implementation: 
        <https://github.com/benedekrozemberczki/ClusterGCN>


    """

    def __init__(self, *graph, n_clusters=None,
                 adj_transformer="normalize_adj", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Cluster Graph Convolutional Networks (ClusterGCN) model.

        This can be instantiated in several ways:

            model = ClusterGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = ClusterGCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        n_clusters: integer. optional
            The number of clusters that the graph being seperated, 
            if not specified (`None`), it will be set to the number 
            of classes automatically. (default :obj: `None`).            
        adj_transformer: string, `transformer`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.transformers`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transformer: string, transformer, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transformers`
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
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        if not n_clusters:
            n_clusters = self.graph.n_classes

        self.n_clusters = n_clusters
        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        attr_matrix = self.attr_transformer(graph.attr_matrix)

        batch_adj, batch_x, self.cluster_member = T.graph_partition(graph.adj_matrix,
                                                                    attr_matrix,
                                                                    n_clusters=self.n_clusters)

        batch_adj = self.adj_transformer(*batch_adj)

        (self.batch_adj, self.batch_x) = T.astensors(batch_adj, batch_x, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[32], activations=['relu'], dropout=0.5,
              l2_norms=[1e-5], lr=0.01, use_bias=False):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.n_attrs],
                      dtype=self.floatx, name='attr_matrix')
            adj = Input(batch_shape=[None, None], dtype=self.floatx,
                        sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None], dtype=self.intx, name='node_index')

            h = x
            for hid, activation, l2_norm in zip(hiddens, activations, l2_norms):
                h = Dropout(rate=dropout)(h)
                h = GraphConvolution(hid, use_bias=use_bias, activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

            h = Dropout(rate=dropout)(h)
            h = GraphConvolution(self.graph.n_classes,
                                 use_bias=use_bias)([h, adj])
            h = Gather()([h, index])

            model = Model(inputs=[x, adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'],
                          experimental_run_tf_function=False)

            self.model = model

    def train_sequence(self, index):
        index = T.asintarr(index)
        mask = T.indices2mask(index, self.graph.n_nodes)
        labels = self.graph.labels

        batch_idx, batch_labels = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            mini_labels = labels[nodes][mini_mask]
            if mini_labels.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_adj.append(self.batch_adj[cluster])
            batch_idx.append(np.where(mini_mask)[0])
            batch_labels.append(mini_labels)

        batch_data = tuple(zip(batch_x, batch_adj, batch_idx))

        sequence = MiniBatchSequence(batch_data, batch_labels, device=self.device)
        return sequence

    def predict(self, index):
        index = T.asintarr(index)
        mask = T.indices2mask(index, self.graph.n_nodes)

        orders_dict = {idx: order for order, idx in enumerate(index)}
        batch_idx, orders = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.n_clusters):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            batch_nodes = np.asarray(nodes)[mini_mask]
            if batch_nodes.size == 0:
                continue
            batch_x.append(self.batch_x[cluster])
            batch_adj.append(self.batch_adj[cluster])
            batch_idx.append(np.where(mini_mask)[0])
            orders.append([orders_dict[n] for n in batch_nodes])

        batch_data = tuple(zip(batch_x, batch_adj, batch_idx))

        logit = np.zeros((index.size, self.graph.n_classes), dtype=self.floatx)
        batch_data = T.astensors(batch_data, device=self.device)

        with tf.device(self.device):
            for order, inputs in zip(orders, batch_data):
                output = self.model.predict_on_batch(inputs)
                logit[order] = output

        return logit
