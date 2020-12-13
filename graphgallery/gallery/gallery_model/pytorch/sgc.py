from graphgallery.nn.layers.pytorch import SGConvolution
from graphgallery.nn.models.pytorch import SGC as pySGC
from graphgallery.gallery import GalleryModel
from graphgallery.sequence import FullBatchSequence

from graphgallery import functional as gf


class SGC(GalleryModel):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def __init__(self,
                 graph,
                 order=2,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Simplifying Graph Convolutional Networks (SGC) model.


        This can be instantiated in the following way:

            model = SGC(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, attributed, labeled graph.
        order: positive integer. optional 
            The power (order) of adjacency matrix. (default :obj: `2`, i.e., 
            math:: A^{2})            
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

        self.order = order

        self.process()

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = self.transform.adj_transform(graph.adj_matrix)
        node_attr = self.transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        X = SGConvolution(order=self.order)([X, A])
        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[],
              activations=[],
              dropout=0.5,
              weight_decay=5e-5,
              lr=0.2,
              use_bias=True):

        self.model = pySGC(self.graph.num_node_attrs,
                           self.graph.num_node_classes,
                           hiddens=hiddens,
                           activations=activations,
                           dropout=dropout,
                           weight_decay=weight_decay,
                           lr=lr,
                           use_bias=use_bias).to(self.device)

    def train_sequence(self, index):
        labels = self.graph.node_label[index]

        X = self.cache.X[index]
        sequence = FullBatchSequence(X, labels, device=self.device)
        return sequence
