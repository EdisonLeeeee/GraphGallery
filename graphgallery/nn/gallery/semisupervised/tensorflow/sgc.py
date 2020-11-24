import tensorflow as tf

from graphgallery.nn.layers.tensorflow import SGConvolution
from graphgallery.nn.models.tensorflow import SGC as tfSGC

from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence

from graphgallery import functional as F


class SGC(SemiSupervisedModel):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def __init__(self, *graph, order=2, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        r"""Create a Simplifying Graph Convolutional Networks (SGC) model.


        This can be instantiated in several ways:

            model = SGC(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SGC(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
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

        self.order = order
        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        node_attr = self.attr_transform(graph.node_attr)

        feature_inputs, structure_inputs = F.astensors(
            node_attr, adj_matrix, device=self.device)

        # To avoid this tensorflow error in large dataset:
        # InvalidArgumentError: Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]
        if self.graph.num_node_attrs * adj_matrix.nnz > 2**31:
            device = "CPU"
        else:
            device = self.device

        with tf.device(device):
            feature_inputs = SGConvolution(order=self.order)(
                [feature_inputs, structure_inputs])

        with tf.device(self.device):
            self.feature_inputs, self.structure_inputs = feature_inputs, structure_inputs

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[], activations=[], dropout=0.5, weight_decay=5e-5, lr=0.2, use_bias=True):

        with tf.device(self.device):
            self.model = tfSGC(self.graph.num_node_attrs, self.graph.num_node_classes, hiddens=hiddens,
                               activations=activations, dropout=dropout, weight_decay=weight_decay,
                               lr=lr, use_bias=use_bias)

    def train_sequence(self, index):
        index = F.astensor(index)
        labels = self.graph.node_label[index]

        feature_inputs = tf.gather(self.feature_inputs, index)
        sequence = FullBatchNodeSequence(feature_inputs, labels, device=self.device)
        return sequence
