from graphgallery.gallery import GalleryModel
from graphgallery.sequence import FullBatchSequence

from graphgallery.nn.models.pytorch import GCN as pyGCN
from graphgallery import functional as gf


class GCN(GalleryModel):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        `Semi-Supervised Classification with Graph Convolutional Networks 
        <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self,
                 graph,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Graph Convolutional Networks (GCN) model.


        This can be instantiated in the following way:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, attributed, labeled graph.
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

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = self.transform.adj_transform(graph.adj_matrix)
        node_attr = self.transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[16],
              activations=['relu'],
              dropout=0.5,
              weight_decay=5e-4,
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

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(
            [self.cache.X, self.cache.A, index],
            labels,
            device=self.device)
        return sequence
