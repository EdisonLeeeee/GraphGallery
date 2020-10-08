import tensorflow as tf

from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength

from graphgallery.nn.models.semisupervised.tf_models.robustgcn import RobustGCN as tfRobustGCN
from graphgallery import transforms as T


class RobustGCN(SemiSupervisedModel):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        `Robust Graph Convolutional Networks Against Adversarial Attacks 
        <https://dl.acm.org/doi/10.1145/3292500.3330851>`
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

    """

    def __init__(self, *graph, adj_transform=T.NormalizeAdj(rate=[-0.5, -1.0]),
                 attr_transform=None, device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Robust Graph Convolutional Networks (RobustGCN or RGCN) model.

        This can be instantiated in several ways:

            model = RobustGCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = RobustGCN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.

        Parameters:
        ----------
            graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
                A sparse, attributed, labeled graph.
            adj_transform: string, `transform`, or None. optional
                How to transform the adjacency matrix. See `graphgallery.transforms`
                (default: :obj:`'normalize_adj'` with normalize rate `-0.5` and `-1`.) 
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

        self.adj_transform = T.get(adj_transform)
        self.attr_transform = T.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = T.astensors(
            attr_matrix, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @EqualVarLength()
    def build(self, hiddens=[64], activations=['relu'], dropout=0.5,
              l2_norm=5e-4, lr=0.01, kl=5e-4, gamma=1., use_bias=False):

        if self.kind == "T":
            with tf.device(self.device):
                self.model = tfRobustGCN(self.graph.n_attrs, self.graph.n_classes,
                                         hiddens=hiddens,
                                         activations=activations,
                                         dropout=dropout, l2_norm=l2_norm,
                                         kl=kl, gamma=gamma,
                                         lr=lr, use_bias=use_bias)
        else:
            raise NotImplementedError

    def train_sequence(self, index):
        
        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, *self.structure_inputs, index], labels, device=self.device)
        return sequence
