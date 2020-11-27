import tensorflow as tf

from graphgallery import functional as F
from graphgallery.gallery import SemiSupervisedModel
from graphgallery.sequence import FullBatchNodeSequence

from graphgallery.nn.models.dgl_tf import GCN as dglGCN


class GCN(SemiSupervisedModel):
    """
        Implementation of Graph Convolutional Networks (GCN). 
        `Semi-Supervised Classification with Graph Convolutional Networks 
        <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

    """

    def __init__(self, *graph, adj_transform="add_selfloops", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        r"""Create a Graph Convolutional Networks (GCN) model.


        This can be instantiated in several ways:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GCN(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'add_selfloops'`, i.e., A = A + I) 
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
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
        kwargs: other custom keyword parameters.
        """
        super().__init__(*graph, device=device, seed=seed, name=name, **kwargs)

        self.adj_transform = F.get(adj_transform)
        self.attr_transform = F.get(attr_transform)
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        node_attr = self.attr_transform(graph.node_attr)

        self.feature_inputs, self.structure_inputs = F.astensors(node_attr, adj_matrix, device=self.device)

    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.5,
              weight_decay=5e-4, lr=0.01, use_bias=True):

        with tf.device(self.device):
            self.model = dglGCN(self.graph.num_node_attrs, self.graph.num_node_classes,
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
        labels = self.graph.node_label[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels,
            device=self.device, escape=type(self.structure_inputs))
        return sequence
