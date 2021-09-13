import uuid
import os.path as osp
import graphgallery as gg


def default_cfg_setup(cfg):
    # Base configs for model
    cfg.task = "Link Prediction"

    # Configs for model preprocessing
    cfg.data = gg.CfgNode()
    cfg.data.graph_transform = None
    cfg.data.edge_transform = None
    cfg.data.adj_transform = None
    cfg.data.attr_transform = None
    cfg.data.label_transform = None
    cfg.data.device = None

    # Configs for model building
    cfg.model = gg.CfgNode()

    # Configs for model training
    cfg.fit = gg.CfgNode()
    cfg.fit.epochs = 100
    cfg.fit.verbose = 1
    cfg.fit.cache_train_data = False
    cfg.fit.cache_val_data = False

    # cfg.fit.EarlyStopping = gg.CfgNode()
    # cfg.fit.EarlyStopping.enabled = False
    # cfg.fit.EarlyStopping.monitor = 'val_ap'
    # cfg.fit.EarlyStopping.verbose = 1
    # cfg.fit.EarlyStopping.mode = "auto"
    # cfg.fit.EarlyStopping.patience = 10
    # cfg.fit.EarlyStopping.restore_best_weights = True
    # cfg.fit.EarlyStopping.baseline = None

    cfg.fit.ModelCheckpoint = gg.CfgNode()
    cfg.fit.ModelCheckpoint.enabled = True
    cfg.fit.ModelCheckpoint.monitor = 'val_ap'
    cfg.fit.ModelCheckpoint.remove_weights = True
    # checkpoint path
    # use `uuid` to avoid duplication
    cfg.fit.ModelCheckpoint.path = osp.join(".",
                                            f"{cfg.name}_checkpoint_{uuid.uuid1().hex[:6]}{gg.file_ext()}")
    cfg.fit.ModelCheckpoint.monitor = 'val_ap'
    cfg.fit.ModelCheckpoint.save_best_only = True
    cfg.fit.ModelCheckpoint.save_weights_only = True
    cfg.fit.ModelCheckpoint.verbose = 0
    cfg.fit.ModelCheckpoint.mode = 'max'

    # cfg.fit.Progbar = gg.CfgNode()
    # cfg.fit.Progbar.width = 20

    # cfg.fit.Logger = gg.CfgNode()
    # cfg.fit.Logger.enabled = False
    # cfg.fit.Logger.name = None
    # cfg.fit.Logger.filepath = None

    # cfg.fit.TensorBoard = gg.CfgNode()
    # cfg.fit.TensorBoard.enabled = False
    # cfg.fit.TensorBoard.log_dir = './logs'
    # cfg.fit.TensorBoard.histogram_freq = 0
    # cfg.fit.TensorBoard.write_graph = True
    # cfg.fit.TensorBoard.write_images = True
    # cfg.fit.TensorBoard.update_freq = 'epoch'

    # Configs for model testing
    cfg.evaluate = gg.CfgNode()
    cfg.evaluate.verbose = 1
    cfg.evaluate.cache_test_data = False

    cfg.evaluate.Progbar = gg.CfgNode()
    cfg.evaluate.Progbar.width = 20

    # Configs for model predicting
    cfg.predict = gg.CfgNode()
    cfg.predict.transform = None
    cfg.predict.cache_predict_data = False
    return cfg
