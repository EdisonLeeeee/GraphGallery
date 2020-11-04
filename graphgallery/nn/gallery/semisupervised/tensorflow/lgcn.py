import tensorflow as tf
import numpy as np

from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence


from graphgallery.nn.models.tensorflow import LGCN as tfLGCN

from graphgallery import functional as F


class LGCN(SemiSupervisedModel):
    """
        Implementation of Large-Scale Learnable Graph Convolutional Networks (LGCN).
        `Large-Scale Learnable Graph Convolutional Networks <https://arxiv.org/abs/1808.03965>`
        Tensorflow 1.x implementation: <https://github.com/divelab/lgcn>
    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Large-Scale Learnable Graph Convolutional Networks (LGCN) model.


        This can be instantiated in several ways:

            model = LGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = LGCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
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

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix).toarray()
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = attr_matrix, adj_matrix

    # @F.EqualVarLength()
    def build(self, hiddens=[32], n_filters=[8, 8], activations=[None, None], dropout=0.8,
              l2_norm=5e-4, lr=0.1, use_bias=False, K=8):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfLGCN(self.graph.n_attrs, self.graph.n_classes,
                                    hiddens=hiddens,
                                    activations=activations,
                                    dropout=dropout, l2_norm=l2_norm,
                                    lr=lr, use_bias=use_bias, K=K)
        else:
            raise NotImplementedError

        self.K = K

    def train_sequence(self, index, batch_size=np.inf):

        mask = F.indices2mask(index, self.graph.n_nodes)
        index = get_indice_graph(self.structure_inputs, index, batch_size)
        while index.size < self.K:
            index = get_indice_graph(self.structure_inputs, index)

        structure_inputs = self.structure_inputs[index][:, index]
        feature_inputs = self.feature_inputs[index]
        mask = mask[index]
        labels = self.graph.labels[index[mask]]

        sequence = FullBatchNodeSequence(
            [feature_inputs, structure_inputs, mask], labels, device=self.device)
        return sequence


def get_indice_graph(adj_matrix, indices, size=np.inf, dropout=0.):
    if dropout > 0.:
        indices = np.random.choice(indices, int(
            indices.size * (1 - dropout)), False)
    neighbors = adj_matrix[indices].sum(axis=0).nonzero()[0]
    if neighbors.size > size - indices.size:
        neighbors = np.random.choice(
            list(neighbors), size - len(indices), False)
    indices = np.union1d(indices, neighbors)
    return indices
