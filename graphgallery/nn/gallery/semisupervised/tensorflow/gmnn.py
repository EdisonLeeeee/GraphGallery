import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.activations import softmax

from graphgallery.nn.layers.tensorflow import GraphConvolution, Gather
from graphgallery.sequence import FullBatchNodeSequence
from graphgallery.nn.gallery import SemiSupervisedModel

from graphgallery import functional as F


class GMNN(SemiSupervisedModel):
    """
        Implementation of Graph Markov Neural Networks (GMNN). 
        `Graph Markov Neural Networks <https://arxiv.org/abs/1905.06214>`
        Pytorch implementation: <https://github.com/DeepGraphLearning/GMNN>


    """

    def __init__(self, *graph, adj_transform="normalize_adj", attr_transform=None,
                 device='cpu:0', seed=None, name=None, **kwargs):
        """Create a Graph Markov Neural Networks (GMNN) model

       This can be instantiated in several ways:

            model = GMNN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

            model = GMNN(adj_matrix, attr_matrix, labels)
                where `adj_matrix` is a 2D Scipy sparse matrix denoting the graph,
                 `attr_matrix` is a 2D Numpy array-like matrix denoting the node 
                 attributes, `labels` is a 1D Numpy array denoting the node labels.


        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph` or a tuple (list) of inputs.
            A sparse, attributed, labeled graph.
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
        self.labels_onehot = self.graph.labels_onehot
        self.custom_objects = {
            'GraphConvolution': GraphConvolution, 'Gather': Gather}
        self.process()

    def process_step(self):
        """
        Process the adjaccelerator.

        Args:
            self: (todo): write your description
        """
        graph = self.graph
        adj_matrix = self.adj_transform(graph.adj_matrix)
        attr_matrix = self.attr_transform(graph.attr_matrix)

        self.feature_inputs, self.structure_inputs = F.astensors(
            attr_matrix, adj_matrix, device=self.device)

    # use decorator to make sure all list arguments have the same length
    @F.EqualVarLength()
    def build(self, hiddens=[16], activations=['relu'], dropout=0.6, l2_norm=5e-4,
              lr=0.05, use_bias=False):
        """
        Builds the model.

        Args:
            self: (todo): write your description
            hiddens: (int): write your description
            activations: (todo): write your description
            dropout: (bool): write your description
            l2_norm: (todo): write your description
            lr: (todo): write your description
            use_bias: (bool): write your description
        """

        with tf.device(self.device):
            x_p = Input(batch_shape=[None, self.graph.n_classes],
                        dtype=self.floatx, name='input_p')
            x_q = Input(batch_shape=[None, self.graph.n_attrs],
                        dtype=self.floatx, name='input_q')
            adj = Input(batch_shape=[None, None],
                        dtype=self.floatx, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],
                          dtype=self.intx, name='node_index')

            def build_GCN(x):
                """
                Builds the model

                Args:
                    x: (todo): write your description
                """
                h = x
                for hidden, activation in zip(hiddens, activations):
                    h = GraphConvolution(hidden, use_bias=use_bias,
                                         activation=activation,
                                         kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                    h = Dropout(rate=dropout)(h)

                h = GraphConvolution(self.graph.n_classes,
                                     use_bias=use_bias)([h, adj])
                h = Gather()([h, index])

                model = Model(inputs=[x, adj, index], outputs=h)
                model.compile(loss=CategoricalCrossentropy(from_logits=True),
                              optimizer=RMSprop(lr=lr), metrics=['accuracy'])
                return model

            # model_p
            model_p = build_GCN(x_p)
            # model_q
            model_q = build_GCN(x_q)

            self.model_p, self.model_q = model_p, model_q
            self.model = self.model_q

    def train(self, idx_train, idx_val=None, pre_train_epochs=100,
              epochs=100, early_stopping=None, verbose=None, save_best=True,
              weight_path=None, as_model=False,
              monitor='val_acc', early_stop_metric='val_loss'):
        """
        Trains the model.

        Args:
            self: (todo): write your description
            idx_train: (array): write your description
            idx_val: (array): write your description
            pre_train_epochs: (bool): write your description
            epochs: (array): write your description
            early_stopping: (str): write your description
            verbose: (bool): write your description
            save_best: (bool): write your description
            weight_path: (str): write your description
            as_model: (str): write your description
            monitor: (todo): write your description
            early_stop_metric: (str): write your description
        """

        histories = []
        index_all = tf.range(self.graph.n_nodes, dtype=self.intx)

        # pre train model_q
        self.model = self.model_q
        history = super().train(idx_train, idx_val, epochs=pre_train_epochs,
                                early_stopping=early_stopping,
                                verbose=verbose, save_best=save_best, weight_path=weight_path, as_model=True,
                                monitor=monitor, early_stop_metric=early_stop_metric)
        histories.append(history)

        label_predict = self.predict(index_all).argmax(1)
        label_predict[idx_train] = self.graph.labels[idx_train]
        label_predict = tf.one_hot(label_predict, depth=self.graph.n_classes)
        # train model_p fitst
        train_sequence = FullBatchNodeSequence([label_predict,
                                                self.structure_inputs, index_all], label_predict,
                                               device=self.device)
        if idx_val is not None:
            val_sequence = FullBatchNodeSequence([label_predict,
                                                  self.structure_inputs, idx_val],
                                                 self.labels_onehot[idx_val],
                                                 device=self.device)
        else:
            val_sequence = None

        self.model = self.model_p
        history = super().train(train_sequence, val_sequence, epochs=epochs,
                                early_stopping=early_stopping,
                                verbose=verbose, save_best=save_best, weight_path=weight_path, as_model=as_model,
                                monitor=monitor, early_stop_metric=early_stop_metric)
        histories.append(history)

        # then train model_q again
        label_predict = self.model.predict_on_batch(
            F.astensors(label_predict, self.structure_inputs, index_all, device=self.device))

        label_predict = softmax(label_predict)
        if tf.is_tensor(label_predict):
            label_predict = label_predict.numpy()

        label_predict[idx_train] = self.labels_onehot[idx_train]

        self.model = self.model_q
        train_sequence = FullBatchNodeSequence([self.feature_inputs,
                                                self.structure_inputs, index_all],
                                               label_predict, device=self.device)
        history = super().train(train_sequence, idx_val, epochs=epochs,
                                early_stopping=early_stopping,
                                verbose=verbose, save_best=save_best,
                                weight_path=weight_path, as_model=as_model,
                                monitor=monitor, early_stop_metric=early_stop_metric)

        histories.append(history)

        return histories

    def train_sequence(self, index):
        """
        Train the model on the given sequence.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """

        # if the graph is changed?
        labels = self.labels_onehot[index]
        sequence = FullBatchNodeSequence(
            [self.feature_inputs, self.structure_inputs, index], labels, device=self.device)
        return sequence
