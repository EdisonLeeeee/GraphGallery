import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tf_layers import GraphAttention, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery.nn.models.semisupervised.th_models.gat import GAT as pyGAT
from graphgallery.nn.models.semisupervised.tf_models.gat import GAT as tfGAT
from graphgallery import transformers as T


class GAT(SemiSupervisedModel):
    """
        Implementation of Graph Attention Networks (GAT).
        `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`
        Tensorflow 1.x implementation: <https://github.com/PetarV-/GAT>
        Pytorch implementation: <https://github.com/Diego999/pyGAT>
        Keras implementation: <https://github.com/danielegrattarola/keras-gat>

    """

    def __init__(self, *graph, adj_transformer="add_selfloops", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Graph Attention Networks (GAT) model.


        This can be instantiated in several ways:

            model = GAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GAT(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.
        Parameters:
        ----------
        graph: graphgallery.data.Graph, or `adj_matrix, attr_matrix and labels` triplets.
            A sparse, attributed, labeled graph.
        adj_transformer: string, transformer, or None. optional
            How to transform the adjacency matrix. (default: :obj:`'normalize_adj'`
            with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
        attr_transformer: string, transformer, or None. optional
            How to transform the node attribute matrix. See `graphgallery.transformers`
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
        kwargs: other customed keyword Parameters.

        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transformer = T.get(adj_transformer)
        self.attr_transformer = T.get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix)
        attr_matrix = self.attr_transformer(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, adj_matrix, device=self.device)

    @EqualVarLength(include=["n_heads"])
    def build(self, hiddens=[8], n_heads=[8], activations=['elu'], 
              dropout=0.6, l2_norms=[5e-4],
              lr=0.01, use_bias=True):

        if self.kind == "P":
            model = pyGAT(self.graph.n_attrs, hiddens, self.graph.n_classes, n_heads=n_heads,
                          activations=activations, l2_norms=l2_norms, dropout=dropout,
                          lr=lr, use_bias=use_bias).to(self.device)
        else:
            with tf.device(self.device):
                model = tfGAT(self.graph.n_attrs, hiddens, self.graph.n_classes, n_heads=n_heads,
                              activations=activations, l2_norms=l2_norms, dropout=dropout,
                              lr=lr, use_bias=use_bias)



        self.model = model

    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)
        return sequence
