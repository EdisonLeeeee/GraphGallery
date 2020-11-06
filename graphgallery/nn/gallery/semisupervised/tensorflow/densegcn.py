import tensorflow as tf


from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence


from graphgallery.nn.models.pytorch import GCN as pyGCN
from graphgallery.nn.models.tensorflow import DenseGCN as tfGCN

from graphgallery import functional as F


class DenseGCN(SemiSupervisedModel):
    """
        Implementation of Dense version of Graph Convolutional Networks (GCN).
        `[`Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x `Sparse version` implementation: <https://github.com/tkipf/gcn>
        Pytorch `Sparse version` implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Dense Graph Convolutional Networks (DenseGCN) model.


        This can be instantiated in several ways:

            model = DenseGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = DenseGCN(adj_matrix, attr_matrix, labels)
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
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix).toarray()
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(
            attr_matrix, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              weight_decay=5e-4, lr=0.01, use_bias=False):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfGCN(self.graph.n_attrs, self.graph.n_classes, hiddens=hiddens,
                                   activations=activations, dropout=dropout, weight_decay=weight_decay,
                                   lr=lr, use_bias=use_bias)
        else:
            self.model = pyGCN(self.graph.n_attrs, self.graph.n_classes, hiddens=hiddens,
                               activations=activations, dropout=dropout, weight_decay=weight_decay,
                               lr=lr, use_bias=use_bias).to(self.device)

    def train_sequence(self, index):

        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)
        return sequence
