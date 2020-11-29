import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GraphConvolution, Gather
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.utils.bvat_utils import kl_divergence_with_logit, entropy_y_x, get_normalized_vector
from graphgallery import functional as gf
from ..obvat import OBVAT


class SimplifiedOBVAT(OBVAT):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        `Batch Virtual Adversarial Training for Graph Convolutional Networks <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """

    def __init__(self,
                 *graph,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 device='cpu:0',
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Simplified OBVAT model.

        This can be instantiated in several ways:

            model = SimplifiedOBVAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SimplifiedOBVAT(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array - like matrix denoting the node
                 attributes, `labels` is a 1D Numpy array denoting the node labels.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple(list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: : obj: `'normalize_adj'` with normalize rate `- 0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default: obj: `None`)
        device: string. optional
            The device where the model is running on. You can specified `CPU` or `GPU`
            for the model. (default:: str: `CPU: 0`, i.e., running on the 0-th `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default: obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: : str: `class.__name__`)
        kwargs: other custom keyword parameters.

        Note:
        ----------
        This is a simplified implementation of `OBVAT`.                
        """
        super().__init__(*graph,
                         adj_transform=adj_transform,
                         attr_transform=attr_transform,
                         device=device,
                         seed=seed,
                         name=name,
                         **kwargs)

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[16],
              activations=['relu'],
              dropout=0.,
              lr=0.01,
              weight_decay=5e-4,
              p1=1.4,
              p2=0.7,
              use_bias=False,
              epsilon=0.01):

        with tf.device(self.device):

            x = Input(batch_shape=[None, self.graph.num_node_attrs],
                      dtype=self.floatx,
                      name='node_attr')
            adj = Input(batch_shape=[None, None],
                        dtype=self.floatx,
                        sparse=True,
                        name='adj_matrix')
            index = Input(batch_shape=[None],
                          dtype=self.intx,
                          name='node_index')

            GCN_layers = []
            for hidden, activation in zip(hiddens, activations):
                GCN_layers.append(
                    GraphConvolution(
                        hidden,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_regularizer=regularizers.l2(weight_decay)))

            GCN_layers.append(
                GraphConvolution(self.graph.num_node_classes,
                                 use_bias=use_bias))

            self.GCN_layers = GCN_layers
            self.dropout = Dropout(rate=dropout)

            logit = self.forward(x, adj)
            output = Gather()([logit, index])

            model = Model(inputs=[x, adj, index], outputs=output)
            model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=Adam(lr=lr),
                          metrics=['accuracy'])

            entropy_loss = entropy_y_x(logit)
            vat_loss = self.virtual_adversarial_loss(x, adj, logit, epsilon)
            model.add_loss(p1 * vat_loss + p2 * entropy_loss)

            self.model = model

    def train_step(self, sequence):
        return super(OBVAT, self).train_step(sequence)

    def virtual_adversarial_loss(self, x, adj, logit, epsilon):
        d = tf.random.normal(
            shape=[self.graph.num_nodes, self.graph.num_node_attrs],
            dtype=self.floatx)

        r_vadv = get_normalized_vector(d) * epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = self.forward(x + r_vadv, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss
