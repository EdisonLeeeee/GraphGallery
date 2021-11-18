import torch
import numpy as np
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import MiniBatchSequence
from graphgallery.gallery import Trainer
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch


@PyTorch.register()
class ClusterGCN(Trainer):
    """
        Implementation of Cluster Graph Convolutional Networks (ClusterGCN).

        `Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
        <https://arxiv.org/abs/1905.07953>`
        Tensorflow 1.x implementation: 
        <https://github.com/google-research/google-research/tree/master/cluster_gcn>
        Pytorch implementation: 
        <https://github.com/benedekrozemberczki/ClusterGCN>
    """

    def data_step(self,
                  adj_transform="normalize_adj",
                  feat_transform=None,
                  num_clusters=10,
                  partition='louvain'):

        graph = self.graph
        batch_adj, batch_feat, cluster_member = gf.graph_partition(
            graph, num_clusters=num_clusters, partition=partition)

        batch_adj = gf.get(adj_transform)(*batch_adj)
        batch_feat = gf.get(feat_transform)(*batch_feat)

        batch_adj, batch_feat = gf.astensors(batch_adj, batch_feat, device=self.data_device)

        # ``A`` and ``X`` and ``cluster_member`` are cached for later use
        self.register_cache(batch_feat=batch_feat, batch_adj=batch_adj,
                            cluster_member=cluster_member)
        # for louvain clustering
        self.num_clusters = len(cluster_member)

    def model_step(self,
                   hids=[32],
                   acts=['relu'],
                   dropout=0.5,
                   bias=False):

        model = models.GCN(self.graph.num_feats,
                           self.graph.num_classes,
                           hids=hids,
                           acts=acts,
                           dropout=dropout,
                           bias=bias)

        return model

    def config_train_data(self, index):
        node_mask = gf.index_to_mask(index, self.graph.num_nodes)
        labels = self.graph.label
        cache = self.cache

        batch_mask, batch_y = [], []
        batch_feat, batch_adj = [], []
        for cluster in range(self.num_clusters):
            nodes = cache.cluster_member[cluster]
            mask = node_mask[nodes]
            y = labels[nodes][mask]
            if y.size == 0:
                continue
            batch_feat.append(cache.batch_feat[cluster])
            batch_adj.append(cache.batch_adj[cluster])
            batch_mask.append(mask)
            batch_y.append(y)

        batch_inputs = tuple(zip(batch_feat, batch_adj))
        sequence = MiniBatchSequence(inputs=batch_inputs,
                                     y=batch_y,
                                     out_index=batch_mask,
                                     device=self.data_device)
        return sequence

    def predict(self, index):
        cache = self.cache

        node_mask = gf.index_to_mask(index, self.graph.num_nodes)
        orders_dict = {idx: order for order, idx in enumerate(index)}
        batch_mask, orders = [], []
        batch_feat, batch_adj = [], []
        for cluster in range(self.num_clusters):
            nodes = cache.cluster_member[cluster]
            mask = node_mask[nodes]
            batch_nodes = np.asarray(nodes)[mask]
            if batch_nodes.size == 0:
                continue
            batch_feat.append(cache.batch_feat[cluster])
            batch_adj.append(cache.batch_adj[cluster])
            batch_mask.append(mask)
            orders.append([orders_dict[n] for n in batch_nodes])

        batch_data = tuple(zip(batch_feat, batch_adj))

        logit = np.zeros((index.size, self.graph.num_classes),
                         dtype=self.floatx)
        batch_data, batch_mask = gf.astensors(batch_data, batch_mask, device=self.data_device)

        model = self.model
        for order, inputs, mask in zip(orders, batch_data, batch_mask):
            output = model.predict_step_on_batch(inputs, out_index=mask)
            logit[order] = output

        return logit

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 0.)
        model = self.model
        return torch.optim.Adam(model.parameters(),
                                weight_decay=weight_decay, lr=lr)
