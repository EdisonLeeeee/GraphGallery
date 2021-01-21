import os
import uuid
import os.path as osp
import graphgallery as gg


def default_cfg(model):
    cfg = gg.CfgNode()
    cfg.name = model.name
    cfg.seed = model.seed
    cfg.device = str(model.device)
    cfg.task = "Node Classification"
    cfg.intx = model.intx
    cfg.floatx = model.floatx
    cfg.boolx = model.boolx
    cfg.backend = getattr(model.backend, "name", None)

    cfg.process = gg.CfgNode()
    cfg.process.graph_transform = None
    cfg.process.adj_transform = None
    cfg.process.attr_transform = None
    cfg.process.label_transform = None

    cfg.model = gg.CfgNode()

    cfg.train = gg.CfgNode()
    cfg.train.epochs = 100
    cfg.train.verbose = 1
    cfg.train.save_best = True

    cfg.train.EarlyStopping = gg.CfgNode()
    cfg.train.EarlyStopping.enabled = False
    cfg.train.EarlyStopping.monitor = 'val_loss'
    cfg.train.EarlyStopping.verbose = 1
    cfg.train.EarlyStopping.mode = "auto"
    cfg.train.EarlyStopping.patience = None

    cfg.train.ModelCheckpoint = gg.CfgNode()
    cfg.train.ModelCheckpoint.enabled = True
    cfg.train.ModelCheckpoint.monitor = 'val_accuracy'
    cfg.train.ModelCheckpoint.remove_weights = True
    # checkpoint path
    # use `uuid` to avoid duplication
    cfg.train.ModelCheckpoint.path = osp.join(".",
                                              f"{cfg.name}_checkpoint_{uuid.uuid1().hex[:6]}{gg.file_ext()}")
    cfg.train.ModelCheckpoint.monitor = 'val_accuracy'
    cfg.train.ModelCheckpoint.save_best_only = True
    cfg.train.ModelCheckpoint.save_weights_only = True
    cfg.train.ModelCheckpoint.vervose = 0

    cfg.train.Progbar = gg.CfgNode()
    cfg.train.Progbar.width = 20

    cfg.test = gg.CfgNode()
    cfg.test.verbose = 1

    cfg.test.Progbar = gg.CfgNode()
    cfg.test.Progbar.width = 20

    cfg.predict = gg.CfgNode()
    cfg.predict.return_logits = True
    return cfg
