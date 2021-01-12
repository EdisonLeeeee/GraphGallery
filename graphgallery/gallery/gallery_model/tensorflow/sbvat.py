import numpy as np
import tensorflow as tf

from graphgallery.sequence import SBVATSampleSequence, FullBatchSequence
from graphgallery.gallery.utils.bvat_utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x

from graphgallery.functional.tensor.tensorflow import gather
from graphgallery import functional as gf
from graphgallery.gallery import TensorFlow
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
        cfg = self.cfg.train
        cfg.p1 = 1.
        cfg.p2 = 1.
        cfg.xi = 1e-6
        cfg.epsilon = 3e-2
        cfg.n_power_iterations = 1
        cfg.num_samples = 50

    def process_step(self,
                     adj_transform="normalize_adj",
                     attr_transform=None,
                     graph_transform=None):

        graph = gf.get(graph_transform)(self.graph)
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A, neighbors=gf.find_4o_nbrs(adj_matrix))

    def builder(self,
                hids=[16],
                acts=['relu'],
                dropout=0.5,
                lr=0.01,
                weight_decay=5e-4,
                use_bias=False,
                use_tfn=True):

        model = get_model("GCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      use_bias=use_bias)

        if use_tfn:
            model.use_tfn()

        return model

    @tf.function
    def train_step(self, sequence):
        model = self.model
        cfg = self.cfg.train

        loss_fn = getattr(model, LOSS)
        metrics = getattr(model, METRICS)
        optimizer = model.optimizer

        with tf.device(self.device):

            for inputs, y, out_weight in sequence:
                x, adj, adv_mask = inputs
                with tf.GradientTape() as tape:
                    logit = model([x, adj], training=True)
                    out = gather(logit, out_weight)
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
        cfg = self.cfg.train
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

    def train_sequence(self, index):
        labels = self.graph.node_label[index]
        sequence = SBVATSampleSequence([self.cache.X, self.cache.A],
                                       labels,
                                       out_weight=index,
                                       neighbors=self.cache.neighbors,
                                       num_samples=self.cfg.train.num_samples,
                                       device=self.device)

        return sequence

    def test_sequence(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_weight=index,
                                     device=self.device)

        return sequence
