import tensorflow as tf
import numpy as np

from graphgallery.gallery import GalleryModel
from graphgallery.sequence import FullBatchSequence

from graphgallery.nn.models.tensorflow import LGCN as tfLGCN

from graphgallery import functional as gf


class LGCN(GalleryModel):
    """
        Implementation of Large-Scale Learnable Graph Convolutional Networks (LGCN).
        `Large-Scale Learnable Graph Convolutional Networks <https://arxiv.org/abs/1808.03965>`
        Tensorflow 1.x implementation: <https://github.com/divelab/lgcn>
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
        r"""Create a Large-Scale Learnable Graph Convolutional Networks (LGCN) model.


        This can be instantiated in the following way:

            model = LGCN(graph)
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
        adj_matrix = self.transform.adj_transform(graph.adj_matrix).toarray()
        node_attr = self.transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

    # @gf.equal()
    def build(self,
              hiddens=[32],
              n_filters=[8, 8],
              activations=[None, None],
              dropout=0.8,
              weight_decay=5e-4,
              lr=0.1,
              use_bias=False,
              K=8):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfLGCN(self.graph.num_node_attrs,
                                    self.graph.num_node_classes,
                                    hiddens=hiddens,
                                    activations=activations,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    lr=lr,
                                    use_bias=use_bias,
                                    K=K)
        else:
            raise NotImplementedError

        self.K = K

    def train_sequence(self, index, batch_size=np.inf):

        mask = gf.index_to_mask(index, self.graph.num_nodes)
        index = get_indice_graph(self.cache.A, index, batch_size)
        while index.size < self.K:
            index = get_indice_graph(self.cache.A, index)

        structure_inputs = self.cache.A[index][:, index]
        feature_inputs = self.cache.X[index]
        mask = mask[index]
        labels = self.graph.node_label[index[mask]]

        sequence = FullBatchSequence(
            [feature_inputs, structure_inputs, mask],
            labels,
            device=self.device)
        return sequence


def get_indice_graph(adj_matrix, indices, size=np.inf, dropout=0.):
    if dropout > 0.:
        indices = np.random.choice(indices, int(indices.size * (1 - dropout)),
                                   False)
    neighbors = adj_matrix[indices].sum(axis=0).nonzero()[0]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(list(neighbors), size - len(indices),
                                     False)
    indices = np.union1d(indices, neighbors)
    return indices
