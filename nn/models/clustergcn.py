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
            
        self.batch_adj, self.batch_features = self._to_tensor([self.batch_adj, self.batch_features])

        
    def build(self, hidden_layers=[32], activations=['relu'], dropout=0.5, learning_rate=0.01, l2_norm=1e-5):
        
        with self.device:
            
            x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')
            adj = Input(batch_shape=[None, None], dtype=tf.float32, sparse=True, name='adj_matrix')
            index = Input(batch_shape=[None],  dtype=tf.int32, name='index')

            h = Dropout(rate=dropout)(x)

            for hid, activation in zip(hidden_layers, activations):
                h = GraphConvolution(hid, activation=activation, kernel_regularizer=regularizers.l2(l2_norm))([h, adj])
                h = Dropout(rate=dropout)(h)

            h = GraphConvolution(self.n_classes)([h, adj])
            h = tf.gather(h, index)
            output = Softmax()(h)

            model = Model(inputs=[x, adj, index], outputs=output)

            model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), 
                          metrics=['accuracy'], experimental_run_tf_function=False)
            
            self.model = model
            self.built = True
            

    def predict(self, index):
        super().predict(index)
        index = self._check_and_convert(index) 
        order_dict = {i: order for order, i in enumerate(index)}
        index = set(index.tolist())
        batch_index, orders = [], []
        for cluster in range(self.n_cluster):
            nodes = set(self.cluster_member[cluster])
            batch_nodes = list(nodes.intersection(index))
            if len(batch_nodes) == 0 :continue
            orders.extend([order_dict[n] for n in batch_nodes])
            batch_index.append(list(map(lambda n: self.mapper[n], batch_nodes)))
        batch_data = tuple(zip(self.batch_features, self.batch_adj, batch_index))
        
        batch_data = self._to_tensor(batch_data)
        logit = []
        with self.device:
            for inputs in batch_data:
                output = self.model.predict_on_batch(inputs)
                logit.append(output)
                
        logit = np.concatenate(logit, axis=0)
        reordered_logit = np.zeros_like(logit)
        reordered_logit[orders] = logit
        return reordered_logit
    
    def train_sequence(self, index):
        if self._is_iterable(index):
            return [self.train_sequence(idx) for idx in index]
        else:
            index = self._check_and_convert(index)
            labels = self.labels[index]
            
            index = set(index.tolist())
            batch_index, batch_labels = [], []
            for cluster in range(self.n_cluster):
                nodes = set(self.cluster_member[cluster])
                batch_nodes = list(nodes.intersection(index))
                batch_index.append(list(map(lambda n: self.mapper[n], batch_nodes)))
                batch_labels.append(self.labels[batch_nodes])

            batch_data = tuple(zip(self.batch_features, self.batch_adj, batch_index))
            with self.device:
                sequence = ClusterMiniBatchSequence(batch_data, batch_labels)
            return sequence
        
        
