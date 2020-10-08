import tensorflow as tf


from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FastGCNBatchSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery.nn.models.semisupervised.tf_models.fastgcn import FastGCN as tfFastGCN

from graphgallery import transforms as T


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
            How to transform the adjacency matrix. See `graphgallery.transforms`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transforms`
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
        self.adj_transform = T.get(adj_transform)
        self.attr_transform = T.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        attr_matrix = adj_matrix @ attr_matrix

        self.feature_inputs, self.structure_inputs = T.astensor(
            attr_matrix, device=self.device), adj_matrix

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[32], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, use_bias=False):

        if self.kind == "T":
            with tf.device(self.device):
                self.model = tfFastGCN(self.graph.n_attrs, self.graph.n_classes,
                                        hiddens=hiddens,
                                        activations=activations,
                                        dropout=dropout, l2_norm=l2_norm,
                                        lr=lr, use_bias=use_bias)
        else:
            raise NotImplementedError

    def train_sequence(self, index):
        
        labels = self.graph.labels[index]
        adj_matrix = self.graph.adj_matrix[index][:, index]
        adj_matrix = self.adj_transform(adj_matrix)

        feature_inputs = tf.gather(self.feature_inputs, index)
        sequence = FastGCNBatchSequence([feature_inputs, adj_matrix], labels,
                                        batch_size=self.batch_size,
                                        rank=self.rank, device=self.device)
        return sequence

    def test_sequence(self, index):
        
        labels = self.graph.labels[index]
        structure_inputs = self.structure_inputs[index]

        sequence = FastGCNBatchSequence([self.feature_inputs, structure_inputs],
                                        labels, batch_size=None, rank=None, device=self.device)  # use full batch
        return sequence
