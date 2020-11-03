import tensorflow as tf


from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence


from graphgallery.nn.models.semisupervised.tensorflow.edgeconv import EdgeGCN as tfEdgeGCN

from graphgallery import functional as F


class EdgeGCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN) -- Edge Convolution version.
        `Semi-Supervised Classification with Graph Convolutional Networks
        <https://arxiv.org/abs/1609.02907>`

        Inspired by: tf_geometric and torch_geometric
        tf_geometric: <https://github.com/CrawlScript/tf_geometric>
        torch_geometric: <https://github.com/rusty1s/pytorch_geometric>

    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Edge Convolution version of Graph Convolutional Networks (EdgeGCN) model.

            This can be instantiated in several ways:

                model = EdgeGCN(graph)
                    with a `graphgallery.data.Graph` instance representing
                    A sparse, attributed, labeled graph.

                model = EdgeGCN(adj_matrix, attr_matrix, labels)
                    where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                     `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                     attributes, `labels` is a 1D Numpy array denoting the node labels.


            Parameters:
            ----------
            graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
                A sparse, attributed, labeled graph.
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

            Note:
            ----------
            The Graph Edge Convolutional implements the operation using message passing 
                framework, i.e., using Tensor `edge index` and `edge weight` of adjacency 
                matrix to aggregate neighbors' message, instead of SparseTensor `adj`.       
            """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)
        edge_index, edge_weight = F.sparse_adj_to_edge(adj_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(
            attr_matrix, (edge_index.T, edge_weight), device=self.device)

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, use_bias=False):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfEdgeGCN(self.graph.n_attrs, self.graph.n_classes,
                                       hiddens=hiddens,
                                       activations=activations,
                                       dropout=dropout, l2_norm=l2_norm,
                                       lr=lr, use_bias=use_bias)
        else:
            raise NotImplementedError

    def train_sequence(self, index):

        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, *self.structure_inputs, index], labels, device=self.device)
        return sequence
