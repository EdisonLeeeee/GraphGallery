import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy

from graphgallery.nn.layers.tensorflow import GCNConv
from graphgallery.data.sequence import FullBatchSequence
from graphgallery.nn.models.tf_keras import TFKeras
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import TensorFlow
from graphgallery.gallery import Trainer


@TensorFlow.register()
class GMNN(Trainer):
    """
        Implementation of Graph Markov Neural Networks (GMNN).
        `Graph Markov Neural Networks <https://arxiv.org/abs/1905.06214>`
        Pytorch implementation: <https://github.com/DeepGraphLearning/GMNN>

    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  attr_transform=None,
                  label_transform="onehot"):
        graph = self.graph
        adj_matrix = gf.get(adj_transform)(graph.adj_matrix)
        node_attr = gf.get(attr_transform)(graph.node_attr)
        label = gf.get(label_transform)(graph.node_label)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.data_device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache(X=X, A=A, Y=label,
                            idx_all=tf.range(graph.num_nodes, dtype=self.intx))

    def model_step(self,
                   hids=[16],
                   acts=['relu'],
                   dropout=0.6,
                   weight_decay=5e-4,
                   lr=0.05,
                   bias=False):

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

        def build_GCN(x):
            h = x
            for hid, act in zip(hids, acts):
                h = GCNConv(
                    hid,
                    use_bias=bias,
                    activation=act,
                    kernel_regularizer=regularizers.l2(weight_decay))(
                        [h, adj])
                h = Dropout(rate=dropout)(h)

            h = GCNConv(self.graph.num_node_classes,
                        use_bias=bias)([h, adj])

            model = TFKeras(inputs=[x, adj], outputs=h)
            model.compile(loss=CategoricalCrossentropy(from_logits=True),
                          optimizer=RMSprop(lr=lr),
                          metrics=['accuracy'])
            return model

        # model_p
        model_p = build_GCN(x_p)
        # model_q
        model_q = build_GCN(x_q)

        model_q.custom_objects = model_p.custom_objects = {
            'GCNConv': GCNConv,
            "TFKeras": TFKeras,
        }
        self.model_p, self.model_q = model_p, model_q
        return model_q

    def fit(self, train_data, val_data=None, **kwargs):
        histories = []

        # pre train model_q
        self.model = self.model_q
        history = super().fit(train_data, val_data,
                              ModelCheckpoint=dict(save_weights_only=False), **kwargs)

        histories.append(history)

        label_predict = self.predict(self.cache.idx_all).argmax(1)
        label_predict[train_data] = self.graph.node_label[train_data]
        label_predict = tf.one_hot(label_predict, depth=self.graph.num_node_classes)
        # train model_p fitst
        train_loader = FullBatchSequence([label_predict, self.cache.A],
                                         label_predict,
                                         device=self.data_device)
        if val_data is not None:
            val_loader = FullBatchSequence([label_predict, self.cache.A],
                                           self.cache.Y[val_data],
                                           out_index=val_data,
                                           device=self.data_device)
        else:
            val_loader = None

        self.model = self.model_p
        history = super().fit(train_loader, val_loader,
                              ModelCheckpoint=dict(save_weights_only=True), **kwargs)
        histories.append(history)
        # then train model_q again
        label_predict = self.model.predict_step_on_batch(x=(label_predict, self.cache.A),
                                                         device=self.data_device)

        if tf.is_tensor(label_predict):
            label_predict = label_predict.numpy()

        label_predict = gf.softmax(label_predict)
        label_predict[train_data] = self.cache.Y[train_data]

        self.model = self.model_q
        train_loader = FullBatchSequence([self.cache.X, self.cache.A],
                                         label_predict,
                                         device=self.data_device)
        history = super().fit(train_loader, val_data,
                              ModelCheckpoint=dict(save_weights_only=True), **kwargs)

        histories.append(history)

        return histories

    def train_loader(self, index):

        labels = self.cache.Y[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_index=index,
                                     device=self.data_device)
        return sequence
