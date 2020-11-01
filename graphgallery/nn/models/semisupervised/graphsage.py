import numpy as np
import tensorflow as tf

from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import SAGEMiniBatchSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery.nn.models.semisupervised.tf_models.graphsage import GraphSAGE as tfGraphSAGE

from graphgallery import transforms as T



class GraphSAGE(SemiSupervisedModel):
    """
        Implementation of SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE). 
        `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`
        Tensorflow 1.x implementation: <https://github.com/williamleif/GraphSAGE>
        Pytorch implementation: <https://github.com/williamleif/graphsage-simple/>
    """

    def __init__(self, *graph, n_samples=(15, 5),
                 adj_transform="neighbor_sampler", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a SAmple and aggreGatE Graph Convolutional Networks (GraphSAGE) model.

        This can be instantiated in several ways:

            model = GraphSAGE(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GraphSAGE(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        n_samples: List of positive integer. optional
            The number of sampled neighbors for each nodes in each layer. 
            (default :obj: `(15, 5)`, i.e., sample `15` first-order neighbors and 
            `5` sencond-order neighbors, and the radius for `GraphSAGE` is `2`)
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.transforms`
            (default: :obj:`'neighbor_sampler'`) 
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
        kwargs: other custom keyword parameters.

        """

        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.n_samples = n_samples
        self.adj_transform = T.get(adj_transform)
        self.attr_transform = T.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        # Dense matrix, shape [n_nodes, max_degree]
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        # pad with a dummy zero vector
        attr_matrix = np.vstack(
            [attr_matrix, np.zeros(attr_matrix.shape[1], dtype=self.floatx)])

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, device=self.device), adj_matrix

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[32], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, use_bias=True, output_normalize=False, aggregator='mean'):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfGraphSAGE(self.graph.n_attrs, self.graph.n_classes,
                                    hiddens=hiddens,
                                    activations=activations,
                                    dropout=dropout, l2_norm=l2_norm,
                                    lr=lr, use_bias=use_bias, aggregator=aggregator,
                                    output_normalize=output_normalize, 
                                    n_samples=self.n_samples)
        else:
            raise NotImplementedError


    def train_sequence(self, index):
        
        labels = self.graph.labels[index]
        sequence = SAGEMiniBatchSequence(
            [self.feature_inputs, self.structure_inputs, index], labels,
            n_samples=self.n_samples, device=self.device)
        return sequence
