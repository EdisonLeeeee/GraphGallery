import scipy.sparse as sp
import tensorflow as tf

from graphgallery.functional.tensor.tensorflow import gather
from graphgallery.sequence import FullBatchSequence
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
class SAT(Trainer):
    def custom_setup(self):
        cfg = self.cfg.fit
        cfg.eps1 = 0.1,
        cfg.eps2 = 0.1,
        cfg.lamb1 = 0.5,
        cfg.lamb2 = 0.5

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  k=35):

        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)

        V, U = sp.linalg.eigsh(adj_matrix, k=k)

        adj_matrix = (U * V) @ U.T
        adj_matrix[adj_matrix < 0] = 0.
        adj_matrix = gf.get(adj_transform)(adj_matrix)

        X, A, U, V = gf.astensors(node_attr,
                                  adj_matrix,
                                  U,
                                  V,
                                  device=self.data_device)
        # ``A`` , ``X`` , U`` and ``V`` are cached for later use
        self.register_cache(X=X, A=A, U=U, V=V)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   weight_decay=5e-4,
                   lr=0.01,
                   bias=False):

        model = get_model("DenseGCN", self.backend)
        model = model(self.graph.num_node_attrs,
                      self.graph.num_node_classes,
                      hids=hids,
                      acts=acts,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr,
                      bias=bias)

        return model

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, sequence):
        (X, A), y, out_index = next(iter(sequence))
        cfg = self.cfg.fit

        U, V = self.cache.U, self.cache.V
        model = self.model
        loss_fn = getattr(model, LOSS)
        metrics = getattr(model, METRICS)
        optimizer = model.optimizer

        with tf.GradientTape() as tape:
            tape.watch([U, V])
            A0 = (U * V) @ tf.transpose(U)
            out = model([X, A0], training=True)
            out = gather(out, out_index)
            loss = loss_fn(y, out)

        U_grad, V_grad = tape.gradient(loss, [U, V])
        U_grad = cfg.eps1 * U_grad / tf.norm(U_grad)
        V_grad = cfg.eps2 * V_grad / tf.norm(V_grad)

        U_hat = U + U_grad
        V_hat = V + V_grad

        with tf.GradientTape() as tape:
            A1 = (U_hat * V) @ tf.transpose(U_hat)
            A2 = (U * V_hat) @ tf.transpose(U)

            out0 = model([X, A0], training=True)
            out0 = gather(out0, out_index)
            out1 = model([X, A1], training=True)
            out1 = gather(out1, out_index)
            out2 = model([X, A2], training=True)
            out2 = gather(out2, out_index)

            loss = loss_fn(y, out0) + tf.reduce_sum(model.losses)
            loss += cfg.lamb1 * loss_fn(y, out1) + cfg.lamb2 * loss_fn(y, out2)
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.update_state(y, out)
            else:
                metrics.update_state(y, out)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        results = [loss] + [
            metric.result() for metric in getattr(metrics, "metrics", metrics)
        ]
        return dict(zip(model.metrics_names, results))

    def train_loader(self, index):
        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
