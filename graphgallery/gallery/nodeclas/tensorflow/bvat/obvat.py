import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GCNConv
from graphgallery.data.sequence import FullBatchSequence
from graphgallery.gallery.nodeclas.tensorflow.bvat.utils import kl_divergence_with_logit, entropy_y_x
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models.tf_keras import TFKeras


@TensorFlow.register()
class OBVAT(Trainer):
    """
        Implementation of optimization-based Batch Virtual Adversarial Training 
        Graph Convolutional Networks (OBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks 
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>

    """

    def custom_setup(self):
        cfg = self.cfg.fit
        cfg.pretrain_epochs = 10
        cfg.stddev = 1e-2

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A)

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False,
                   p1=1.4,
                   p2=0.7):

        x = Input(batch_shape=[None, self.graph.num_node_attrs], dtype=self.floatx, name='node_attr')
        adj = Input(batch_shape=[None, None], dtype=self.floatx, sparse=True, name='adj_matrix')

        GCN_layers = []
        for hid, act in zip(hids, acts):
            GCN_layers.append(
                GCNConv(
                    hid,
                    activation=act,
                    use_bias=bias,
                    kernel_regularizer=regularizers.l2(weight_decay)))

        GCN_layers.append(
            GCNConv(self.graph.num_node_classes,
                    use_bias=bias))
        self.GCN_layers = GCN_layers
        self.dropout = Dropout(rate=dropout)

        h = self.forward(x, adj)

        model = TFKeras(inputs=[x, adj], outputs=h)
        model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        self.r_vadv = tf.Variable(TruncatedNormal(stddev=0.01)(shape=[self.graph.num_nodes,
                                                                      self.graph.num_node_attrs]), name="r_vadv")
        entropy_loss = entropy_y_x(h)
        vat_loss = self.virtual_adversarial_loss(x, adj, h)
        model.add_loss(p1 * vat_loss + p2 * entropy_loss)

        self.adv_optimizer = Adam(lr=lr / 10.)

        return model

    def virtual_adversarial_loss(self, x, adj, logit):

        adv_x = x + self.r_vadv
        logit_p = tf.stop_gradient(logit)
        logit_m = self.forward(adv_x, adj)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss

    def forward(self, x, adj):
        h = x
        for layer in self.GCN_layers:
            h = self.dropout(h)
            h = layer([h, adj])
        return h

    @tf.function
    def pretrain(self, x, adj, r_vadv):
        cfg = self.cfg.fit
        with tf.device(self.device):
            optimizer = self.adv_optimizer
            r_vadv.assign(TruncatedNormal(stddev=cfg.stddev)(shape=tf.shape(r_vadv)))
            for _ in range(cfg.pretrain_epochs):
                with tf.GradientTape() as tape:
                    rnorm = tf.nn.l2_loss(r_vadv)
                    logit = self.forward(x, adj)
                    vloss = self.virtual_adversarial_loss(x, adj, logit)
                    loss = rnorm - vloss
                gradient = tape.gradient(loss, r_vadv)
                optimizer.apply_gradients(zip([gradient], [r_vadv]))

    def train_step(self, sequence):
        self.pretrain(self.cache.X, self.cache.A, self.r_vadv)
        return super().train_step(sequence)

    def train_loader(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
