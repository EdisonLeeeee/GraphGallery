import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from graphgallery.nn.layers import GraphConvolution
from graphgallery.mapper import ClusterMiniBatchSequence, FullBatchNodeSequence
from graphgallery.utils import partition_graph, Bunch
from .base import SupervisedModel

class ClusterGCN(SupervisedModel):
    
    def __init__(self, adj, features, labels, graph=None, n_cluster=None,
                 normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None):
    
        super().__init__(adj, features, labels, device=device, seed=seed)
        
        if n_cluster is None:
            n_cluster = self.n_classes
            
        self.n_cluster = n_cluster
        self.normalize_rate = normalize_rate
        self.normalize_features = normalize_features
        self.preprocess(adj, features, graph)
        
    def preprocess(self, adj, features, graph=None):  
        
        if self.normalize_features:
            features = self._normalize_features(features)
            
        if graph is None:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
            
        (self.batch_adj, self.batch_features, self.batch_labels, 
         self.cluster_member, self.mapper) = partition_graph(adj, features, self.labels, graph, 
                                                             n_cluster=self.n_cluster)
        
        if self.normalize_rate is not None:
            self.batch_adj = self._normalize_adj(self.batch_adj, self.normalize_rate)
            
        with self.device:
            self.batch_adj, self.batch_features = self._to_tensor([self.batch_adj, self.batch_features])

        
    def build(self, hidden_layers=[32], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=1e-5):
        
        with self.device:
            
            x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')
            adj = Input(batch_shape=[None, None], dtype=tf.float32, sparse=True, name='adj_matrix')
            mask = Input(batch_shape=[None],  dtype=tf.bool, name='mask')

            h = Dropout(rate=dropout)(x)

            for hid, activation in zip(hidden_layers, activations):
                h = GraphConvolution(hid, activation=activation, kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = GraphConvolution(self.n_classes)([h, adj])
            h = tf.boolean_mask(h, mask)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, mask], outputs=output)

            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), 
                          metrics=['accuracy'], experimental_run_tf_function=False)
            
            self.model = model
            self.built = True
            

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index) 
        mask = self._sample_mask(index)
        
        order_dict = {idx: order for order, idx in enumerate(index)}
        batch_mask, orders = [], []
        batch_features, batch_adj = [], []
        for cluster in range(self.n_cluster):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            batch_nodes = np.asarray(nodes)[mini_mask]
            if batch_nodes.size == 0: continue
            batch_features.append(self.batch_features[cluster])
            batch_adj.append(self.batch_adj[cluster])            
            batch_mask.append(mini_mask)
            orders.append([order_dict[n] for n in batch_nodes])
            
        batch_data = tuple(zip(batch_features, batch_adj, batch_mask))
        
        logit = np.zeros((index.size, self.n_classes), dtype='float32')
        with self.device:
            batch_data = self._to_tensor(batch_data)
            for order, inputs in zip(orders, batch_data):
                output = self.model.predict_on_batch(inputs)
                logit[order] = output
                
        return logit

        
    def train_sequence(self, index):
        index = self._check_and_convert(index)
        mask = self._sample_mask(index)
        labels = self.labels

        batch_mask, batch_labels = [], []
        batch_features, batch_adj = [], []
        for cluster in range(self.n_cluster):
            nodes = self.cluster_member[cluster]
            mini_mask = mask[nodes]
            mini_labels = labels[nodes][mini_mask]
            if mini_labels.size==0: continue
            batch_features.append(self.batch_features[cluster])
            batch_adj.append(self.batch_adj[cluster])
            batch_mask.append(mini_mask)
            batch_labels.append(mini_labels)

        batch_data = tuple(zip(batch_features, batch_adj, batch_mask))
        with self.device:
            sequence = ClusterMiniBatchSequence(batch_data, batch_labels)
        return sequence
