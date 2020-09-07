import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import DenseConvolution, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.shape import EqualVarLength
from graphgallery import astensors, asintarr, normalize_x, normalize_adj, Bunch


class DenseGCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN).
        `[`Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, graph, norm_adj=-0.5, norm_x=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Dense Graph Convolutional Network.

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


        Note:
        ----------
        The input adjacency matrix will be transformed into dense one,
            which needs more memory usage. It's not recommended to use
            this model in a large-scale dataset.                
        """

        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)

        self.norm_adj = norm_adj
        self.norm_x = norm_x
        self.preprocess(adj, x)
        # set to `False` to suggest the Dense inputs
        self.sparse = False

    def preprocess(self, adj, x):
        super().preprocess(adj, x)
        adj, x = self.adj, self.x

        if self.norm_adj:
            adj = normalize_adj(adj, self.norm_adj)

        if sp.isspmatrix(adj):
            adj = adj.toarray()

        if self.norm_x:
            x = normalize_x(x, norm=self.norm_x)

        with tf.device(self.device):
            self.x_norm, self.adj_norm = astensors([x, adj])

    def build(self, hiddens=[16], activations=['relu'], dropouts=[0.5], l2_norms=[5e-4],
              lr=0.01, use_bias=False):

        ############# Record paras ###########
        local_paras = locals()
        local_paras.pop('self')
        paras = Bunch(**local_paras)
        hiddens, activations, dropouts, l2_norms = set_equal_in_length(hiddens, activations, dropouts, l2_norms)
        paras.update(Bunch(hiddens=hiddens, activations=activations, dropouts=dropouts, l2_norms=l2_norms))
        # update all parameters
        self.paras.update(paras)
        self.model_paras.update(paras)
        ######################################

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.n_attrs], dtype=self.floatx, name='attr_matrix')
            adj = Input(batch_shape=[None, None], dtype=self.floatx, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=self.intx, name='node_index')

            h = x
            for hid, activation, dropout, l2_norm in zip(hiddens, activations, dropouts, l2_norms):
                h = DenseConvolution(hid, use_bias=use_bias,
                                     activation=activation,
                                     kernel_regularizer=regularizers.l2(l2_norm))([h, adj])

                h = Dropout(rate=dropout)(h)

            h = DenseConvolution(self.n_classes, use_bias=use_bias)([h, adj])
            h = Gather()([h, index])

            model = Model(inputs=[x, adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])
            self.model = model

    def train_sequence(self, index):
        index = asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence([self.x_norm, self.adj_norm, index], labels)
        return sequence

    def predict(self, index):
        super().predict(index)
        index = asintarr(index)
        with tf.device(self.device):
            index = astensors(index)
            logit = self.model.predict_on_batch([self.x_norm, self.adj_norm, index])

        if tf.is_tensor(logit):
            logit = logit.numpy()
        return logit
