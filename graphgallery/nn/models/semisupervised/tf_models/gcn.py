import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import GraphConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.shape import EqualVarLength
from graphgallery.transformers import get
from graphgallery import astensors, asintarr


class GCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, *graph, adj_transformer="normalize_adj", attr_transformer=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Graph Convolutional Networks (GCN) model.

        Parameters:
        ----------
            graph: graphgallery.data.Graph
                A sparse, attributed, labeled graph.
            norm_adj: float scalar. optional 
                The normalize rate for adjacency matrix `adj`. (default: :obj:`-0.5`, 
                i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) 
            norm_x: string. optional
                How to normalize the node attribute matrix. See `graphgallery.normalize_x`
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

        self.adj_transformer = get(adj_transformer)
        self.attr_transformer = get(attr_transformer)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transformer(graph.adj_matrix)
        attr_matrix = self.attr_transformer(graph.attr_matrix)
        
        with tf.device(self.device):
            self.feature_inputs, self.structure_inputs = astensors(
                attr_matrix, adj_matrix)

    @EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5],
              l2_norms=[5e-4], lr=0.01, use_bias=False):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.n_attrs], dtype=self.floatx, name='attr_matrix')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None], dtype=self.intx, name='node_index')

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = GraphConvolution(hid, use_bias=use_bias,
                                     activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

                h = Dropout(rate=dropout)(h)

            h = GraphConvolution(self.graph.n_classes,
                                 use_bias=use_bias)([h, adj])
            h = Gather()([h, index])

            model = Model(inputs=[x, adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])

            self.model = model

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.graph.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence(
                [self.feature_inputs, self.structure_inputs, index], labels)
        return sequence
