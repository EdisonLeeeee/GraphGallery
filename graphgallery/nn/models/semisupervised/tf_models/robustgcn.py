import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers import GaussionConvolution_F, GaussionConvolution_D, Sample, Gather
from graphgallery.nn.models import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.decorators import EqualVarLength
from graphgallery import transformers as T
from graphgallery.transformers import NormalizeAdj


class RobustGCN(SemiSupervisedModel):
    """
        Implementation of Robust Graph Convolutional Networks (RobustGCN). 
        `Robust Graph Convolutional Networks Against Adversarial Attacks 
        <https://dl.acm.org/doi/10.1145/3292500.3330851>`
        Tensorflow 1.x implementation: <https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip>

    """

    def __init__(self, *graph, adj_transformer=T.NormalizeAdj(rate=[-0.5, -1.0]),
                 attr_transformer=None, device='cpu:0', seed=None, name=None, **kwargs):
        """Creat a Robust Graph Convolutional Networks (RobustGCN or RGCN) model.

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
            adj_transformer: string, `transformer`, or None. optional
                How to transform the adjacency matrix. See `graphgallery.transformers`
                (default: :obj:`'normalize_adj'` with normalize rate `-0.5` and `-1`.) 
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

        with tf.device(self.device):
            self.feature_inputs, self.structure_inputs = T.astensors(
                attr_matrix, adj_matrix)

    @EqualVarLength()
    def build(self, hiddens=[64], activations=['relu'], use_bias=False, dropouts=[0.5],
              l2_norms=[5e-4], lr=0.01, kl=5e-4, gamma=1.):

        with tf.device(self.device):
            x = Input(batch_shape=[None, self.graph.n_attrs],
                      dtype=self.floatx, name='attr_matrix')
            adj = [Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix_1'),
                   Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix_2')]
            index = Input(batch_shape=[None],
                          dtype=self.intx, name='node_index')

            h = x
            mean, var = GaussionConvolution_F(hiddens[0], gamma=gamma,
                                              use_bias=use_bias,
                                              activation=activations[0],
                                              kernel_regularizer=regularizers.l2(l2_norms[0]))([h, *adj])
            if kl:
                KL_divergence = 0.5 * \
                    tf.reduce_mean(tf.math.square(mean) + var -
                                   tf.math.log(1e-8 + var) - 1, axis=1)
                KL_divergence = tf.reduce_sum(KL_divergence)

                # KL loss
                kl_loss = kl * KL_divergence

            # additional layers (usually unnecessay)
            for hid, activation, dropout, l2_norm in zip(hiddens[1:], activations[1:], dropouts[1:], l2_norms[1:]):

                mean, var = GaussionConvolution_D(
                    hid, gamma=gamma, use_bias=use_bias, activation=activation)([mean, var, *adj])
                mean = Dropout(rate=dropout)(mean)
                var = Dropout(rate=dropout)(var)

            mean, var = GaussionConvolution_D(
                self.graph.n_classes, gamma=gamma, use_bias=use_bias)([mean, var, *adj])
            h = Sample(seed=self.seed)([mean, var])
            h = Gather()([h, index])

            model = Model(inputs=[x, *adj, index], outputs=h)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr), metrics=['accuracy'])

            if kl:
                model.add_loss(kl_loss)
            self.model = model

    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence(
                [self.x_norm, *self.adj_norm, index], labels)
        return sequence

    def train_sequence(self, index):
        index = T.asintarr(index)
        labels = self.graph.labels[index]
        with tf.device(self.device):
            sequence = FullBatchNodeSequence(
                [self.feature_inputs, *self.structure_inputs, index], labels)
        return sequence
