import numpy as np
import tensorflow as tf

from graphgallery.gallery import GalleryModel
from graphgallery.sequence import SBVATSampleSequence, FullBatchNodeSequence
from graphgallery.utils.bvat_utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x

from graphgallery import functional as F

from graphgallery.nn.models.tensorflow import GCN as tfGCN


class SBVAT(GalleryModel):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """
    def __init__(self,
                 *graph,
                 n_samples=50,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 device='cpu:0',
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT) model.

         This can be instantiated in several ways:

            model = SBVAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = SBVAT(adj_matrix, node_attr, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `node_attr` is a 2D Numpy array-like matrix denoting the node
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
        n_samples (Positive integer, optional):
            The number of sampled subset nodes in the graph where the length of the
            shortest path between them is at least `4`. (default :obj: `50`)
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
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
        self.n_samples = n_samples
        self.process()

    def process_step(self):
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        node_attr = self.attr_transform(graph.node_attr)
        self.neighbors = F.find_4o_nbrs(adj_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(
            node_attr, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @F.equal
    def build(self,
              hiddens=[16],
              activations=['relu'],
              dropout=0.5,
              lr=0.01,
              weight_decay=5e-4,
              use_bias=False,
              p1=1.,
              p2=1.,
              n_power_iterations=1,
              epsilon=0.03,
              xi=1e-6):

        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model = tfGCN(self.graph.num_node_attrs,
                                   self.graph.num_node_classes,
                                   hiddens=hiddens,
                                   activations=activations,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   lr=lr,
                                   use_bias=use_bias)
                self.index_all = tf.range(self.graph.num_nodes,
                                          dtype=self.intx)
        else:
            raise NotImplementedError

        self.p1 = p1  # Alpha
        self.p2 = p2  # Beta
        self.xi = xi  # Small constant for finite difference
        # Norm length for (virtual) adversarial training
        self.epsilon = epsilon
        self.n_power_iterations = n_power_iterations  # Number of power iterations

    @tf.function
    def train_step(self, sequence):
        model = self.model
        metric = model.metrics[0]
        loss_fn = model.loss
        optimizer = model.optimizer

        with tf.device(self.device):
            metric.reset_states()

            for inputs, labels in sequence:
                x, adj, index, adv_mask = inputs
                with tf.GradientTape() as tape:
                    logit = model([x, adj, self.index_all], training=True)
                    output = tf.gather(logit, index)
                    loss = loss_fn(labels, output)
                    entropy_loss = entropy_y_x(logit)
                    vat_loss = self.virtual_adversarial_loss(x,
                                                             adj,
                                                             logit=logit,
                                                             adv_mask=adv_mask)
                    loss += self.p1 * vat_loss + self.p2 * entropy_loss

                    metric.update_state(labels, output)

                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

            return loss, metric.result()

    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)
        model = self.model
        for _ in range(self.n_power_iterations):
            d = get_normalized_vector(d) * self.xi
            logit_p = logit
            with tf.GradientTape() as tape:
                tape.watch(d)
                logit_m = model([x + d, adj, self.index_all], training=True)
                dist = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
            grad = tape.gradient(dist, d)
            d = tf.stop_gradient(grad)

        r_vadv = get_normalized_vector(d) * self.epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = model([x + r_vadv, adj, self.index_all])
        loss = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
        return loss

    def train_sequence(self, index):

        labels = self.graph.node_label[index]

        sequence = SBVATSampleSequence(
            [self.feature_inputs, self.structure_inputs, index],
            labels,
            neighbors=self.neighbors,
            n_samples=self.n_samples,
            device=self.device)

        return sequence

    def test_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index],
            labels,
            device=self.device)

        return sequence
