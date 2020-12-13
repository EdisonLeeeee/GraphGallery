import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.activations import softmax

from graphgallery.nn.layers.tensorflow import GraphConvolution, Gather
from graphgallery.sequence import FullBatchSequence
from graphgallery.gallery import GalleryModel

from graphgallery import functional as gf
from graphgallery.nn.models import TFKeras


class GMNN(GalleryModel):
    """
        Implementation of Graph Markov Neural Networks (GMNN).
        `Graph Markov Neural Networks <https://arxiv.org/abs/1905.06214>`
        Pytorch implementation: <https://github.com/DeepGraphLearning/GMNN>


    """

    def __init__(self,
                 graph,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a Graph Markov Neural Networks (GMNN) model

        This can be instantiated in the following way:

            model = GMNN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, attributed, labeled graph.
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default :obj: `None`)
        graph_transform: string, `transform` or None. optional
            How to transform the graph, by default None.
        device: string. optional
            The device where the model is running on.
            You can specified ``CPU``, ``GPU`` or ``cuda``
            for the model. (default: :str: `cpu`, i.e., running on the `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.

        """

        super().__init__(graph, device=device, seed=seed, name=name,
                         adj_transform=adj_transform,
                         attr_transform=attr_transform,
                         graph_transform=graph_transform,
                         **kwargs)
        self.register_cache("label_onehot", gf.onehot(self.graph.node_label))
        self.custom_objects = {
            'GraphConvolution': GraphConvolution,
            'Gather': Gather
        }

        self.process()

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = self.transform.adj_transform(graph.adj_matrix)
        node_attr = self.transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

    # use decorator to make sure all list arguments have the same length
    @ gf.equal()
    def build(self,
              hiddens=[16],
              activations=['relu'],
              dropout=0.6,
              weight_decay=5e-4,
              lr=0.05,
              use_bias=False):

        with tf.device(self.device):
            x_p = Input(batch_shape=[None, self.graph.num_node_classes],
                        dtype=self.floatx,
                        name='input_p')
            x_q = Input(batch_shape=[None, self.graph.num_node_attrs],
                        dtype=self.floatx,
                        name='input_q')
            adj = Input(batch_shape=[None, None],
                        dtype=self.floatx,
                        sparse=True,
                        name='adj_matrix')
            index = Input(batch_shape=[None],
                          dtype=self.intx,
                          name='node_index')

            def build_GCN(x):
                h = x
                for hidden, activation in zip(hiddens, activations):
                    h = GraphConvolution(
                        hidden,
                        use_bias=use_bias,
                        activation=activation,
                        kernel_regularizer=regularizers.l2(weight_decay))(
                            [h, adj])
                    h = Dropout(rate=dropout)(h)

                h = GraphConvolution(self.graph.num_node_classes,
                                     use_bias=use_bias)([h, adj])
                h = Gather()([h, index])

                model = TFKeras(inputs=[x, adj, index], outputs=h)
                model.compile(loss=CategoricalCrossentropy(from_logits=True),
                              optimizer=RMSprop(lr=lr),
                              metrics=['accuracy'])
                return model

            # model_p
            model_p = build_GCN(x_p)
            # model_q
            model_q = build_GCN(x_q)

            self.model_p, self.model_q = model_p, model_q
            self.model = self.model_q

    def train(self,
              idx_train,
              idx_val=None,
              pre_train_epochs=100,
              epochs=100,
              early_stopping=None,
              verbose=1,
              save_best=True,
              ckpt_path=None,
              as_model=False,
              monitor='val_accuracy',
              early_stop_metric='val_loss'):

        histories = []
        index_all = tf.range(self.graph.num_nodes, dtype=self.intx)

        # pre train model_q
        self.model = self.model_q
        history = super().train(idx_train,
                                idx_val,
                                epochs=pre_train_epochs,
                                early_stopping=early_stopping,
                                verbose=verbose,
                                save_best=save_best,
                                ckpt_path=ckpt_path,
                                as_model=True,
                                monitor=monitor,
                                early_stop_metric=early_stop_metric)
        histories.append(history)

        label_predict = self.predict(index_all).argmax(1)
        label_predict[idx_train] = self.graph.node_label[idx_train]
        label_predict = tf.one_hot(label_predict,
                                   depth=self.graph.num_node_classes)
        # train model_p fitst
        train_sequence = FullBatchSequence(
            [label_predict, self.cache.A, index_all],
            label_predict,
            device=self.device)
        if idx_val is not None:
            val_sequence = FullBatchSequence(
                [label_predict, self.cache.A, idx_val],
                self.cache.label_onehot[idx_val],
                device=self.device)
        else:
            val_sequence = None

        self.model = self.model_p
        history = super().train(train_sequence,
                                val_sequence,
                                epochs=epochs,
                                early_stopping=early_stopping,
                                verbose=verbose,
                                save_best=save_best,
                                ckpt_path=ckpt_path,
                                as_model=as_model,
                                monitor=monitor,
                                early_stop_metric=early_stop_metric)
        histories.append(history)

        # then train model_q again
        label_predict = self.model.predict_on_batch(
            gf.astensors(label_predict,
                         self.cache.A,
                         index_all,
                         device=self.device))

        label_predict = softmax(label_predict)
        if tf.is_tensor(label_predict):
            label_predict = label_predict.numpy()

        label_predict[idx_train] = self.cache.label_onehot[idx_train]

        self.model = self.model_q
        train_sequence = FullBatchSequence(
            [self.cache.X, self.cache.A, index_all],
            label_predict,
            device=self.device)
        history = super().train(train_sequence,
                                idx_val,
                                epochs=epochs,
                                early_stopping=early_stopping,
                                verbose=verbose,
                                save_best=save_best,
                                ckpt_path=ckpt_path,
                                as_model=as_model,
                                monitor=monitor,
                                early_stop_metric=early_stop_metric)

        histories.append(history)

        return histories

    def train_sequence(self, index):

        # if the graph is changed?
        labels = self.cache.label_onehot[index]
        sequence = FullBatchSequence(
            [self.cache.X, self.cache.A, index],
            labels,
            device=self.device)
        return sequence
