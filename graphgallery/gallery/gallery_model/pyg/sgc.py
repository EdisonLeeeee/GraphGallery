from graphgallery.gallery import GalleryModel
from graphgallery.nn.models.pyg import SGC as pygSGC
from graphgallery.sequence import FullBatchNodeSequence
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
                 adj_transform=None,
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
            How to transform the graph, by default, the graph transform is used
            before the other transform unless specify ``graph_first=False``
        device: string. optional
            The device where the model is running on. You can specified `CPU` or `GPU` 
            for the model. (default: :str: `cpu`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional 
            Used in combination with `tf.random.set_seed` & `np.random.seed` 
            & `random.seed` to create a reproducible sequence of tensors across 
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: keyword parameters for transform, 
            e.g., ``graph_first`` argument indicating the graph transform is
            used at the first or last, by default at the first.
        """
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

        self.order = order
        self.adj_transform = gf.get(adj_transform)
        self.attr_transform = gf.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        node_attr = self.attr_transform(graph.node_attr)

        self.feature_inputs, self.structure_inputs = gf.astensors(
            node_attr, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[],
              activations=[],
              dropout=0.,
              weight_decay=5e-5,
              lr=0.2,
              use_bias=True):

        self.model = pygSGC(self.graph.num_node_attrs,
                            self.graph.num_node_classes,
                            hiddens=hiddens,
                            K=self.order,
                            activations=activations,
                            dropout=dropout,
                            weight_decay=weight_decay,
                            lr=lr,
                            use_bias=use_bias).to(self.device)

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, *self.structure_inputs, index],
            labels,
            device=self.device)
        return sequence
