# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model, Input
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras import regularizers

# from nn.layers import SGConvolution
# from .base import SupervisedModel
# from mapper import FullBatchNodeSequence
# from utils import metis_clustering

# class ClusterSGC(SupervisedModel):
    
#     def __init__(self, adj, features, data, graph, n_cluster=None, normalize_rate=-0.5, normalize_features=True, device='CPU:0', seed=None):
        
#         super().__init__(data, device=device, seed=seed)
        
#         if n_cluster is None:
#             n_cluster = data.n_classes
            
#         if normalize_features:
#             features = features / (features.sum(1, keepdims=True) + 1e-10)
            
#         parts = metis_clustering(graph, n_cluster)
        
        
# #         if not normalize_rate is None:
# #             adj = self._normalize_adj(adj, normalize_rate)
    
#         cluster_member = [[] for _ in range(n_cluster)]
#         for node_index, part in enumerate(parts):
#             cluster_member[part].append(node_index)
            
#         x = np.zeros_like(features)
#         SGC_layer = SGConvolution()
        
#         for cluster in range(n_cluster):
#             nodes = cluster_member[cluster]
#             mini_adj = adj[nodes][:, nodes]
#             mini_features = features[nodes]
#             if not normalize_rate is None:
#                 mini_adj = self._normalize_adj(mini_adj, normalize_rate)  
#             mini_features, mini_adj = self._to_tensor([mini_features, mini_adj])
#             with self.device:
#                 mini_x = SGC_layer([mini_features, mini_adj])
#             x[nodes] = mini_x.numpy()
            
#         x = self._to_tensor(x)
            
#         mask_train, mask_val, mask_test = self._to_tensor([data.mask_train, 
#                                                                data.mask_val, 
#                                                                data.mask_test])
            
#         x_train = tf.boolean_mask(x, mask_train)
#         x_val = tf.boolean_mask(x, mask_val)
#         x_test = tf.boolean_mask(x, mask_test)

#         with self.device:

#             self.data_train = FullBatchNodeSequence(x_train, data.y_train)
#             self.data_val = FullBatchNodeSequence(x_val, data.y_val)
#             self.data_test = FullBatchNodeSequence(x_test, data.y_test)
                
#             self.x = x
        
#     def build(self, learning_rate=0.2, l2_norm=5e-5):
        
#         with self.device:
            
#             x = Input(batch_shape=[None, self.n_features], dtype=tf.float32, name='features')

#             output = layers.Dense(self.n_classes, activation='softmax', kernel_regularizer=regularizers.l2(l2_norm))(x)
            
#             model = Model(inputs=x, outputs=output)
#             model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

#             self.model = model
#             self.built = True

#     def predict(self, idx):
        
#         if not self.built:
#             raise RuntimeError('You must compile your model before training/testing/predicting. Use `model.build()`.')
            
#         self._check(idx)
        
#         with self.device:
#             x = tf.gather(self.x, idx)
#             logit = self.model.predict_on_batch(x=x)
        
#         return logit.numpy()
