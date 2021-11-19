import torch
import numpy as np
import graphgallery.nn.models.pytorch as models
from graphgallery.data.sequence import MiniBatchSequence
from graphgallery.gallery import Trainer
from graphgallery import functional as gf
from graphgallery.gallery.nodeclas import PyTorch
from torch.utils.data import DataLoader, Dataset


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

        assert partition in {'metis', 'random', 'louvain'}

        graph = self.graph
        batch_adj, batch_feat, cluster_member = gf.graph_partition(
            graph, num_clusters=num_clusters, partition=partition)

        batch_adj = gf.get(adj_transform)(*batch_adj)
        batch_feat = gf.get(feat_transform)(*batch_feat)

        batch_adj, batch_feat = gf.astensors(batch_adj, batch_feat, device=self.data_device)

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
            nodes = np.array(cache.cluster_member[cluster])
            mask = node_mask[nodes]
            y = labels[nodes][mask]
            if len(y) == 0:
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

    def config_predict_data(self, index):
        node_mask = gf.index_to_mask(index, self.graph.num_nodes)
        labels = self.graph.label
        cache = self.cache

        batch_mask, batch_y = [], []
        batch_feat, batch_adj = [], []
        batch_nodes = []
        for cluster in range(self.num_clusters):
            nodes = np.array(cache.cluster_member[cluster])
            mask = node_mask[nodes]
            y = labels[nodes][mask]
            if len(y) == 0:
                continue
            batch_feat.append(cache.batch_feat[cluster])
            batch_adj.append(cache.batch_adj[cluster])
            batch_mask.append(mask)
            batch_y.append(y)
            batch_nodes.append(nodes[mask])

        batch_inputs = tuple(zip(batch_feat, batch_adj))
        sequence = MiniBatchSequence(inputs=batch_inputs,
                                     y=batch_y,
                                     out_index=batch_mask,
                                     node_ids=batch_nodes,
                                     device=self.data_device)
        return sequence

    @torch.no_grad()
    def predict_step(self, dataloader):
        model = self.model
        model.eval()
        outs = []
        ids = []
        callbacks = self.callbacks
        for epoch, batch in enumerate(dataloader):
            callbacks.on_predict_batch_begin(epoch)
            x, _, out_mask, node_ids = self.unravel_batch(batch)
            x = self.to_device(x)
            out = model(*x)
            if out_mask is not None:
                out = out[out_mask]
            outs.append(out)
            ids.append(node_ids)
            callbacks.on_predict_batch_end(epoch)
        return torch.cat(outs, dim=0), torch.cat(ids, dim=0)

    def predict(self, predict_data=None,
                transform=torch.nn.Softmax(dim=-1)):

        indices = gf.astensor(predict_data).view(-1)
        mapper = torch.zeros(self.graph.num_nodes).long()
        mapper[indices] = torch.arange(indices.size(0))

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(predict_data, (DataLoader, Dataset)):
            predict_data = self.config_predict_data(predict_data)

        out, node_ids = self.predict_step(predict_data)
        out[mapper[node_ids.cpu()]] = out.clone()

        out = out.squeeze()
        if transform is not None:
            out = transform(out)
        return out.cpu()

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 0.)
        return torch.optim.Adam(self.model.parameters(),
                                weight_decay=weight_decay, lr=lr)
