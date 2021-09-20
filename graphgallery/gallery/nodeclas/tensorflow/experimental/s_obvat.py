import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GCNConv
from graphgallery.gallery.nodeclas.tensorflow.bvat.utils import kl_divergence_with_logit, entropy_y_x, get_normalized_vector
from graphgallery.nn.models.tf_keras import TFKeras
from graphgallery.gallery.nodeclas import TensorFlow

from ..bvat.obvat import OBVAT


@TensorFlow.register()
class SimplifiedOBVAT(OBVAT):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training  Graph Convolutional Networks (OBVAT). 
        `Batch Virtual Adversarial Training for Graph Convolutional Networks <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.,
                   lr=0.01,
                   weight_decay=5e-4,
                   bias=False,
                   p1=1.4,
                   p2=0.7,
                   epsilon=0.01):

        x = Input(batch_shape=[None, self.graph.num_node_attrs],
                  dtype=self.floatx,
                  name='node_attr')
        adj = Input(batch_shape=[None, None],
                    dtype=self.floatx,
                    sparse=True,
                    name='adj_matrix')

        GCN_layers = []
        for hid, act in zip(hids, acts):
            GCN_layers.append(
                GCNConv(
                    hid,
                    activation=act,
                    bias=bias,
                    kernel_regularizer=regularizers.l2(weight_decay)))

        GCN_layers.append(
            GCNConv(self.graph.num_node_classes,
                    bias=bias))

        self.GCN_layers = GCN_layers
        self.dropout = Dropout(rate=dropout)

        h = self.forward(x, adj)

        model = TFKeras(inputs=[x, adj], outputs=h)
        model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        entropy_loss = entropy_y_x(h)
        vat_loss = self.virtual_adversarial_loss(x, adj, h, epsilon)
        model.add_loss(p1 * vat_loss + p2 * entropy_loss)

        return model

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
