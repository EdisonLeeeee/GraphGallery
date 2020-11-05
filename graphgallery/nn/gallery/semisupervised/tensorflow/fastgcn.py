import tensorflow as tf


from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FastGCNBatchSequence


from graphgallery.nn.models.tensorflow import FastGCN as tfFastGCN

from graphgallery import functional as F


class FastGCN(SemiSupervisedModel):
    """
        Implementation of Fast Graph Convolutional Networks (FastGCN).
        `FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling 
        <https://arxiv.org/abs/1801.10247>`
        Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def __init__(self, *graph, batch_size=256, rank=100,
                 adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Fast Graph Convolutional Networks (FastGCN) model.


        This can be instantiated in several ways:

            model = FastGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = FastGCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        batch_size (Positive integer, optional):
            Batch size for the training nodes. (default :int: `256`)
        rank (Positive integer, optional):
            The selected nodes for each batch nodes, `rank` must be smaller than
            `batch_size`. (default :int: `100`)
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

        self.rank = rank
        self.batch_size = batch_size
        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        """
        Process the adjacian.

        Args:
            self: (todo): write your description
        """
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        attr_matrix = adj_matrix @ attr_matrix

        self.feature_inputs, self.structure_inputs = F.astensor(
            attr_matrix, device=self.device), adj_matrix

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[32], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, use_bias=False):
        """
        Builds the graph.

        Args:
            self: (todo): write your description
            hiddens: (int): write your description
            activations: (todo): write your description
            dropout: (bool): write your description
            l2_norm: (todo): write your description
            lr: (todo): write your description
            use_bias: (bool): write your description
        """

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfFastGCN(self.graph.n_attrs, self.graph.n_classes,
                                       hiddens=hiddens,
                                       activations=activations,
                                       dropout=dropout, l2_norm=l2_norm,
                                       lr=lr, use_bias=use_bias)
        else:
            raise NotImplementedError

    def train_sequence(self, index):
        """
        Assigns the model.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """

        labels = self.graph.labels[index]
        adj_matrix = self.graph.adj_matrix[index][:, index]
        adj_matrix = self.adj_transform(adj_matrix)

        feature_inputs = tf.gather(self.feature_inputs, index)
        sequence = FastGCNBatchSequence([feature_inputs, adj_matrix], labels,
                                        batch_size=self.batch_size,
                                        rank=self.rank, device=self.device)
        return sequence

    def test_sequence(self, index):
        """
        Takes a batch of the target sequence.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """

        labels = self.graph.labels[index]
        structure_inputs = self.structure_inputs[index]

        sequence = FastGCNBatchSequence([self.feature_inputs, structure_inputs],
                                        labels, batch_size=None, rank=None, device=self.device)  # use full batch
        return sequence
