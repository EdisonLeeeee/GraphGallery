import numpy as np
import tensorflow as tf

from graphgallery.gallery import GalleryModel
from graphgallery.sequence import MiniBatchSequence

from graphgallery.nn.models.tensorflow import GCN as tfGCN
from graphgallery import functional as F


class ClusterGCN(GalleryModel):
    """
        Implementation of Cluster Graph Convolutional Networks (ClusterGCN).

        `Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
        <https://arxiv.org/abs/1905.07953>`
        Tensorflow 1.x implementation: 
        <https://github.com/google-research/google-research/tree/master/cluster_gcn>
        Pytorch implementation: 
        <https://github.com/benedekrozemberczki/ClusterGCN>


    """
    def __init__(self,
                 *graph,
                 n_clusters=None,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 device='cpu:0',
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Cluster Graph Convolutional Networks (ClusterGCN) model.

        This can be instantiated in several ways:

            model = ClusterGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = ClusterGCN(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        n_clusters: integer. optional
            The number of clusters that the graph being seperated, 
            if not specified (`None`), it will be set to the number 
            of classes automatically. (default :obj: `None`).            
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
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
        kwargs: other custom keyword parameters.
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        if not n_clusters:
            n_clusters = self.graph.num_node_classes

        self.n_clusters = n_clusters
        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        node_attr = self.attr_transform(graph.node_attr)

        batch_adj, batch_x, self.cluster_member = F.graph_partition(
            graph.adj_matrix, node_attr, n_clusters=self.n_clusters)

        batch_adj = self.adj_transform(*batch_adj)

        (self.batch_adj, self.batch_x) = F.astensors(batch_adj,
                                                     batch_x,
                                                     device=self.device)

    # use decorator to make sure all list arguments have the same length
    @F.equal
    def build(self,
              hiddens=[32],
              activations=['relu'],
              dropout=0.5,
              weight_decay=0.,
              lr=0.01,
              use_bias=False):

        with tf.device(self.device):
            self.model = tfGCN(self.graph.num_node_attrs,
                               self.graph.num_node_classes,
                               hiddens=hiddens,
                               activations=activations,
                               dropout=dropout,
                               weight_decay=weight_decay,
                               lr=lr,
                               use_bias=use_bias,
                               experimental_run_tf_function=False)

    def train_sequence(self, index):

        mask = F.indices2mask(index, self.graph.num_nodes)
        labels = self.graph.node_label

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

        sequence = MiniBatchSequence(batch_data,
                                     batch_labels,
                                     device=self.device)
        return sequence

    def predict(self, index):

        mask = F.indices2mask(index, self.graph.num_nodes)

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

        logit = np.zeros((index.size, self.graph.num_node_classes),
                         dtype=self.floatx)
        batch_data = F.astensors(batch_data, device=self.device)

        model = self.model
        with tf.device(self.device):
            for order, inputs in zip(orders, batch_data):
                output = model.predict_on_batch(inputs)
                logit[order] = output

        return logit
