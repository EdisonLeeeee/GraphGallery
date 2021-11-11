import tensorflow as tf

from graphgallery.data.sequence import SBVATSampleSequence, FullBatchSequence
from graphgallery.gallery.nodeclas.tensorflow.bvat.utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x

from graphgallery.functional.tensor.tensorflow import gather
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model

from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
    METRICS = "compiled_metrics"
    LOSS = "compiled_loss"
else:
    METRICS = "metrics"
    LOSS = "loss"


@TensorFlow.register()
class SBVAT(Trainer):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>
    """

    def custom_setup(self):
        cfg = self.cfg.fit
        cfg.p1 = 1.
        cfg.p2 = 1.
        cfg.xi = 1e-6
        cfg.epsilon = 3e-2
        cfg.n_power_iterations = 1
        cfg.sizes = 50

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  sizes=50):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        attr_matrix = gf.get(attr_transform)(graph.attr_matrix)

        X, A = gf.astensors(attr_matrix, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A, neighbors=gf.find_4o_nbrs(adj_matrix))

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.5,
                   lr=0.01,
                   weight_decay=5e-4,
                   bias=False):

        model = get_model("GCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    @tf.function
    def train_step(self, sequence):
        model = self.model
        cfg = self.cfg.fit

        loss_fn = getattr(model, LOSS)
        metrics = getattr(model, METRICS)
        optimizer = model.optimizer

        with tf.device(self.device):

            for inputs, y, out_index in sequence:
                x, adj, adv_mask = inputs
                with tf.GradientTape() as tape:
                    logit = model([x, adj], training=True)
                    out = gather(logit, out_index)
                    loss = loss_fn(y, out)
                    entropy_loss = entropy_y_x(logit)
                    vat_loss = self.virtual_adversarial_loss(x,
                                                             adj,
                                                             logit=logit,
                                                             adv_mask=adv_mask)
                    loss += cfg.p1 * vat_loss + cfg.p2 * entropy_loss

                    if isinstance(metrics, list):
                        for metric in metrics:
                            metric.update_state(y, out)
                    else:
                        metrics.update_state(y, out)

                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

            results = [loss] + [
                metric.result()
                for metric in getattr(metrics, "metrics", metrics)
            ]
            return dict(zip(model.metrics_names, results))

    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        cfg = self.cfg.fit
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)
        model = self.model
        for _ in range(cfg.n_power_iterations):
            d = get_normalized_vector(d) * cfg.xi
            logit_p = logit
            with tf.GradientTape() as tape:
                tape.watch(d)
                logit_m = model([x + d, adj])
                dist = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
            grad = tape.gradient(dist, d)
            d = tf.stop_gradient(grad)

        r_vadv = get_normalized_vector(d) * cfg.epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = model([x + r_vadv, adj], training=True)
        loss = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
        return loss

    def train_loader(self, index):
        labels = self.graph.label[index]
        sequence = SBVATSampleSequence(inputs=[self.cache.X, self.cache.A],
                                       neighbors=self.cache.neighbors,
                                       y=labels,
                                       out_index=index,
                                       sizes=self.cfg.data.sizes,
                                       device=self.data_device)

        return sequence

    def test_loader(self, index):
        labels = self.graph.label[index]
        sequence = FullBatchSequence(inputs=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_index=index,
                                     device=self.data_device)

        return sequence
