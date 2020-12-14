import torch
import numpy as np
from graphgallery.gallery import GalleryModel
from graphgallery.sequence import MiniBatchSequence

from graphgallery.nn.models.pytorch import GCN as pyGCN
from graphgallery import functional as gf


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
                 graph,
                 n_clusters=None,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Cluster Graph Convolutional Networks (ClusterGCN) model.

        This can be instantiated in the following way:

            model = ClusterGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
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
        graph_transform: string, `transform` or None. optional
            How to transform the graph, by default None.
        device: string. optional
            The device where the model is running on.
            You can specified ``CPU``, ``GPU`` or ``cuda``
            for the model. (default: :str: `cpu`, i.e., running on the `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """
        super().__init__(graph, device=device, seed=seed, name=name,
                         adj_transform=adj_transform,
                         attr_transform=attr_transform,
                         graph_transform=graph_transform,
                         **kwargs)
        self.register_cache("n_clusters", n_clusters)
        self.process()

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        graph.node_attr = self.transform.attr_transform(graph.node_attr)

        batch_adj, batch_x, cluster_member = gf.graph_partition(
            graph, n_clusters=self.cache.n_clusters)

        batch_adj = self.transform.adj_transform(*batch_adj)
        batch_adj, batch_x = gf.astensors(batch_adj, batch_x, device=self.device)

        # ``A`` and ``X`` and ``cluster_member`` are cached for later use
        self.register_cache("batch_x", batch_x)
        self.register_cache("batch_adj", batch_adj)
        self.register_cache("cluster_member", cluster_member)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[32],
              activations=['relu'],
              dropout=0.5,
              weight_decay=0.,
              lr=0.01,
              use_bias=False):

        self.model = pyGCN(self.graph.num_node_attrs,
                           self.graph.num_node_classes,
                           hiddens=hiddens,
                           activations=activations,
                           dropout=dropout,
                           weight_decay=weight_decay,
                           lr=lr,
                           use_bias=use_bias).to(self.device)

    def train_sequence(self, index):

        mask = gf.index_to_mask(index, self.graph.num_nodes)
        labels = self.graph.node_label

        batch_idx, batch_labels = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.cache.n_clusters):
            nodes = self.cache.cluster_member[cluster]
            batch_mask = mask[nodes]
            y = labels[nodes][batch_mask]
            if y.size == 0:
                continue
            batch_x.append(self.cache.batch_x[cluster])
            batch_adj.append(self.cache.batch_adj[cluster])
            batch_idx.append(np.where(batch_mask)[0])
            batch_labels.append(y)

        batch_data = tuple(zip(batch_x, batch_adj, batch_idx))

        sequence = MiniBatchSequence(batch_data,
                                     batch_labels,
                                     device=self.device)
        return sequence

    def predict(self, index):

        mask = gf.index_to_mask(index, self.graph.num_nodes)

        orders_dict = {idx: order for order, idx in enumerate(index)}
        batch_idx, orders = [], []
        batch_x, batch_adj = [], []
        for cluster in range(self.cache.n_clusters):
            nodes = self.cache.cluster_member[cluster]
            batch_mask = mask[nodes]
            batch_nodes = np.asarray(nodes)[batch_mask]
            if batch_nodes.size == 0:
                continue
            batch_x.append(self.cache.batch_x[cluster])
            batch_adj.append(self.cache.batch_adj[cluster])
            batch_idx.append(np.where(batch_mask)[0])
            orders.append([orders_dict[n] for n in batch_nodes])

        batch_data = tuple(zip(batch_x, batch_adj, batch_idx))

        logit = np.zeros((index.size, self.graph.num_node_classes),
                         dtype=self.floatx)
        batch_data = gf.astensors(batch_data, device=self.device)

        model = self.model
        model.eval()
        with torch.no_grad():
            for order, inputs in zip(orders, batch_data):
                output = model(inputs).detach().cpu().numpy()
                logit[order] = output

        return logit
