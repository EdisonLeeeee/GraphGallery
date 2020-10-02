import tensorflow as tf

from graphgallery.nn.layers.tf_layers import SGConvolution as tfSGConvolution
from graphgallery.nn.layers.th_layers import SGConvolution as pySGConvolution

from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery.nn.models.semisupervised.th_models.sgc import SGC as pySGC
from graphgallery.nn.models.semisupervised.tf_models.sgc import SGC as tfSGC

from graphgallery import transforms as T


class SGC(SemiSupervisedModel):
    """
        Implementation of Simplifying Graph Convolutional Networks (SGC). 
        `Simplifying Graph Convolutional Networks <https://arxiv.org/abs/1902.07153>`
        Pytorch implementation: <https://github.com/Tiiiger/SGC>

    """

    def __init__(self, *graph, order=2, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Simplifying Graph Convolutional Networks (SGC) model.


        This can be instantiated in several ways:

            model = SGC(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SGC(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        order: positive integer. optional 
            The power (order) of adjacency matrix. (default :obj: `2`, i.e., 
            math:: A^{2})            
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
        kwargs: other customized keyword Parameters.
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.order = order
        self.adj_transform = T.get(adj_transform)
        self.attr_transform = T.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        feature_inputs, structure_inputs = T.astensors(
            attr_matrix, adj_matrix, device=self.device)

        if self.kind == "T":
            # To avoid this tensorflow error in large dataset:
            # InvalidArgumentError: Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]
            if self.graph.n_attrs * adj_matrix.nnz > 2**31:
                device = "CPU"
            else:
                device = self.device

            with tf.device(device):
                feature_inputs = tfSGConvolution(order=self.order)(
                    [feature_inputs, structure_inputs])

            with tf.device(self.device):
                self.feature_inputs, self.structure_inputs = feature_inputs, structure_inputs
        else:
            feature_inputs = pySGConvolution(order=self.order)(
                [feature_inputs, structure_inputs])
            self.feature_inputs, self.structure_inputs = feature_inputs, structure_inputs

            

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[], activations=[], dropout=0.5, l2_norm=5e-5, lr=0.2, use_bias=True):

        if self.kind == "T":
            with tf.device(self.device):
                self.model = tfSGC(self.graph.n_attrs, self.graph.n_classes, hiddens=hiddens,
                              activations=activations, dropout=dropout, l2_norm=l2_norm,
                              lr=lr, use_bias=use_bias)
        else:
            self.model = pySGC(self.graph.n_attrs, self.graph.n_classes, hiddens=hiddens,
                          activations=activations, dropout=dropout, l2_norm=l2_norm,
                               lr=lr, use_bias=use_bias).to(self.device)


    def train_sequence(self, index):
        index = T.astensor(index)
        labels = self.graph.labels[index]
        
        if self.kind == "T":
            feature_inputs = tf.gather(self.feature_inputs, index)
        else:
            feature_inputs = self.feature_inputs[index]
        sequence = FullBatchNodeSequence(feature_inputs, labels, device=self.device)
        return sequence
