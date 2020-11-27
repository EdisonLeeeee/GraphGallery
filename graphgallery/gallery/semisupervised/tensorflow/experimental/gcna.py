import tensorflow as tf


from graphgallery.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence


from graphgallery.nn.models.tensorflow import GCNA as tfGCNA
from graphgallery import functional as F
from ..gcn import GCN


class GCNA(GCN):
    """
    GCN + node node attribute matrix

    Implementation of Graph Convolutional Networks(GCN) concated with node node attribute matrix.
    `Semi - Supervised Classification with Graph Convolutional Networks 
    <https://arxiv.org/abs/1609.02907>`
    GCN Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
    GCN Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        r"""Create a Graph Convolutional Networks(GCN) model 
            concated with node node attribute matrix (GCNA).

        This can be instantiated in several ways:

            model = GCNA(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GCNA(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array - like matrix denoting the node
                 attributes, `labels` is a 1D Numpy array denoting the node labels.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple(list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default:: obj: `'normalize_adj'` with normalize rate `- 0.5`.
            i.e., math: : \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default: obj: `None`)
        device: string. optional
            The device where the model is running on. You can specified `CPU` or `GPU`
            for the model. (default: : str: `CPU: 0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default: obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default:: str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """
        super().__init__(*graph,
                         adj_transform=adj_transform, attr_transform=attr_transform,
                         device=device, seed=seed, name=name, **kwargs)

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              weight_decay=5e-4, lr=0.01, use_bias=False):

        with tf.device(self.device):
            self.model = tfGCNA(self.graph.num_node_attrs, self.graph.num_node_classes, hiddens=hiddens,
                                activations=activations, dropout=dropout, weight_decay=weight_decay,
                                lr=lr, use_bias=use_bias)
