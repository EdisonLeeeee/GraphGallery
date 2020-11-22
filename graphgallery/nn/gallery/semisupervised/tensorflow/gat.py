import tensorflow as tf

from graphgallery.nn.layers.tensorflow import GraphAttention, Gather
from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence

from graphgallery.nn.models.tensorflow import GAT as tfGAT
from graphgallery import functional as F


class GAT(SemiSupervisedModel):
    """
        Implementation of Graph Attention Networks (GAT).
        `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`
        Tensorflow 1.x implementation: <https://github.com/PetarV-/GAT>
        Pytorch implementation: <https://github.com/Diego999/pyGAT>
        Keras implementation: <https://github.com/danielegrattarola/keras-gat>

    """

    def __init__(self, *graph, adj_transform="add_selfloops", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        r"""Create a Graph Attention Networks (GAT) model.


        This can be instantiated in several ways:

            model = GAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GAT(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.
        Parameters:
        ----------
        graph: graphgallery.data.Graph, or `adj_matrix, node_attr and labels` triplets.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. 
            (default: :obj:`'add_selfloops'`, i.e., A = A + I) 
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

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        node_attr = self.attr_transform(graph.node_attr)

        self.feature_inputs, self.structure_inputs = F.astensors(
            node_attr, adj_matrix, device=self.device)

    @F.EqualVarLength(include=["n_heads"])
    def build(self, hiddens=[8], n_heads=[8], activations=['elu'],
              dropout=0.6, weight_decay=5e-4,
              lr=0.01, use_bias=True):

        with tf.device(self.device):
            self.model = tfGAT(self.graph.num_node_attrs, self.graph.num_node_classes,
                               hiddens=hiddens, n_heads=n_heads,
                               activations=activations,
                               dropout=dropout,
                               weight_decay=weight_decay,
                               lr=lr, use_bias=use_bias)

    def train_sequence(self, index):

        labels = self.graph.node_labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)
        return sequence
