import numpy as np
import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora', root="~/GraphData/datasets/", verbose=False, transform="standardize")
graph = data.graph
splits = data.split_nodes(random_state=15)

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.DenseGCN(device=device, seed=123).setup_graph(graph).build(hids=32)
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=200)

################### Attacker model ############################
attacker = gg.attack.untargeted.PGD(graph, device=device, seed=None).process(
    trainer, splits.train_nodes, unlabeled_nodes=splits.test_nodes)
attacker.attack(0.05, CW_loss=0, C=100)

################### Victim model ############################
# This is a white-box attack
# Before attack
original_result = trainer.evaluate(splits.test_nodes)

# After attack
# reprocess after the graph has changed
trainer.setup_graph(attacker.g)  # important!
perturbed_result = trainer.evaluate(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)
"""original prediction 84.91%
perturbed prediction 80.08%
The accuracy has gone down 4.83%
"""
