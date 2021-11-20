import torch
from typing import Callable
from graphgallery.nn.metrics import Accuracy

from graphgallery.gallery.trainer import Trainer


class NodeClasTrainer(Trainer):
    def config_loss(self) -> Callable:
        return torch.nn.CrossEntropyLoss()

    def config_metrics(self) -> Callable:
        return Accuracy()

    def _test_predict(self, index):
        logit = self.predict(index).cpu().numpy()
        predict_class = logit.argmax(-1)
        labels = self.graph.label[index]
        return (predict_class == labels).mean()

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
