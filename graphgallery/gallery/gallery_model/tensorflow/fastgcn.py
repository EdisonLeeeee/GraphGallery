import tensorflow as tf

from graphgallery.gallery import GalleryModel
from graphgallery.sequence import FastGCNBatchSequence

from graphgallery.nn.models.tensorflow import FastGCN as tfFastGCN

from graphgallery import functional as gf


class FastGCN(GalleryModel):
    """
        Implementation of Fast Graph Convolutional Networks (FastGCN).
        `FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling 
        <https://arxiv.org/abs/1801.10247>`
        Tensorflow 1.x implementation: <https://github.com/matenure/FastGCN>

    """

    def __init__(self,
                 graph,
                 batch_size=256,
                 rank=100,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Fast Graph Convolutional Networks (FastGCN) model.


        This can be instantiated in the following way:

            model = FastGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
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

        self.register_cache("rank", rank)
        self.register_cache("batch_size", batch_size)

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = self.transform.adj_transform(graph.adj_matrix)
        node_attr = self.transform.attr_transform(graph.node_attr)
        node_attr = adj_matrix @ node_attr

        X, A = gf.astensor(node_attr, device=self.device), adj_matrix

        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[32],
              activations=['relu'],
              dropout=0.5,
              weight_decay=5e-4,
              lr=0.01,
              use_bias=False):

        with tf.device(self.device):
            self.model = tfFastGCN(self.graph.num_node_attrs,
                                   self.graph.num_node_classes,
                                   hiddens=hiddens,
                                   activations=activations,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   lr=lr,
                                   use_bias=use_bias)

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        adj_matrix = self.graph.adj_matrix[index][:, index]
        adj_matrix = self.cache.adj_transform(adj_matrix)

        X = tf.gather(self.cache.X, index)
        sequence = FastGCNBatchSequence([X, adj_matrix],
                                        labels,
                                        batch_size=self.batch_size,
                                        rank=self.rank,
                                        device=self.device)
        return sequence

    def test_sequence(self, index):

        labels = self.graph.node_label[index]
        A = self.cache.A[index]

        sequence = FastGCNBatchSequence(
            [self.cache.X, A],
            labels,
            batch_size=None,
            rank=None,
            device=self.device)  # use full batch
        return sequence
