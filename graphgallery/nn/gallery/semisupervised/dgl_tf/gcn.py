import tensorflow as tf

from graphgallery import functional as F
from graphgallery.nn.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence

from graphgallery.nn.models.dgl_tf import GCN as dglGCN


class GCN(SemiSupervisedModel):
    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(attr_matrix, adj_matrix, device=self.device)

    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              weight_decay=5e-4, lr=0.01, use_bias=True):

        with tf.device(self.device):
            self.model = dglGCN(self.graph.n_attrs, self.graph.n_classes,
                                hiddens=hiddens, activations=activations, dropout=dropout,
                                weight_decay=weight_decay, lr=lr, use_bias=use_bias)

    def train_step(self, sequence):

        model = self.model
        weight_decay = getattr(model, "weight_decay", 0.)
        loss_fn = model.loss
        optimizer = model.optimizer
        metric = model.metric
        model.reset_metrics()
        metric.reset_states()

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                for inputs, labels in sequence:
                    logits = model(inputs)
                    metric.update_state(labels, logits)
                    loss = loss_fn(labels, logits)
                    for weight in model.trainable_weights:
                        loss += weight_decay * tf.nn.l2_loss(weight)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss.numpy().item(), metric.result().numpy().item()

    def test_step(self, sequence):

        model = self.model
        weight_decay = getattr(model, "weight_decay", 0)
        loss_fn = model.loss
        metric = model.metric
        model.reset_metrics()
        metric.reset_states()

        with tf.device(self.device):
            for inputs, labels in sequence:
                logits = model(inputs, training=False)
                metric.update_state(labels, logits)
                loss = loss_fn(labels, logits)
                for weight in model.trainable_weights:
                    loss += weight_decay * tf.nn.l2_loss(weight)

        return loss.numpy().item(), metric.result().numpy().item()

    def train_sequence(self, index):
        labels = self.graph.labels[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels,
            device=self.device, escape=type(self.structure_inputs))
        return sequence
