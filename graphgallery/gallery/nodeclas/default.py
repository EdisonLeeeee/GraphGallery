import uuid
import os.path as osp
import graphgallery as gg


def default_cfg(model):
    # Base configs for model
    cfg = gg.CfgNode()
    cfg.name = model.name
    cfg.seed = model.seed
    cfg.device = str(model.device)
    cfg.task = "Node Classification"
    cfg.intx = model.intx
    cfg.floatx = model.floatx
    cfg.boolx = model.boolx
    cfg.backend = getattr(model.backend, "name", None)

    # Configs for model preprocessing
    cfg.process = gg.CfgNode()
    cfg.process.graph_transform = None
    cfg.process.edge_transform = None
    cfg.process.adj_transform = None
    cfg.process.attr_transform = None
    cfg.process.label_transform = None

    # Configs for model building
    cfg.model = gg.CfgNode()
    cfg.model.build_from_other_model = False

    # Configs for model training
    cfg.fit = gg.CfgNode()
    cfg.fit.epochs = 100
    cfg.fit.verbose = 1
    cfg.fit.cache_train_data = True
    cfg.fit.cache_val_data = True

    cfg.fit.EarlyStopping = gg.CfgNode()
    cfg.fit.EarlyStopping.enabled = False
    cfg.fit.EarlyStopping.monitor = 'val_loss'
    cfg.fit.EarlyStopping.verbose = 1
    cfg.fit.EarlyStopping.mode = "auto"
    cfg.fit.EarlyStopping.patience = 10
    cfg.fit.EarlyStopping.restore_best_weights = True
    cfg.fit.EarlyStopping.baseline = None

    cfg.fit.ModelCheckpoint = gg.CfgNode()
    cfg.fit.ModelCheckpoint.enabled = True
    cfg.fit.ModelCheckpoint.monitor = 'val_accuracy'
    cfg.fit.ModelCheckpoint.remove_weights = True
    # checkpoint path
    # use `uuid` to avoid duplication
    cfg.fit.ModelCheckpoint.path = osp.join(".",
                                            f"{cfg.name}_checkpoint_{uuid.uuid1().hex[:6]}{gg.file_ext()}")
    cfg.fit.ModelCheckpoint.monitor = 'val_accuracy'
    cfg.fit.ModelCheckpoint.save_best_only = True
    cfg.fit.ModelCheckpoint.save_weights_only = True
    cfg.fit.ModelCheckpoint.vervose = 0

    cfg.fit.Progbar = gg.CfgNode()
    cfg.fit.Progbar.width = 20

    cfg.fit.TerminateOnNaN = gg.CfgNode()
    cfg.fit.TerminateOnNaN.enabled = False

    cfg.fit.TensorBoard = gg.CfgNode()
    cfg.fit.TensorBoard.enabled = False
    cfg.fit.TensorBoard.log_dir = './logs'
    cfg.fit.TensorBoard.histogram_freq = 0
    cfg.fit.TensorBoard.write_graph = True
    cfg.fit.TensorBoard.write_images = True
    cfg.fit.TensorBoard.update_freq = 'epoch'

    # Configs for model testing
    cfg.evaluate = gg.CfgNode()
    cfg.evaluate.verbose = 1
    cfg.evaluate.cache_test_data = True

    cfg.evaluate.Progbar = gg.CfgNode()
    cfg.evaluate.Progbar.width = 20

    # Configs for model predicting
    cfg.predict = gg.CfgNode()
    cfg.predict.return_logits = True
    return cfg
