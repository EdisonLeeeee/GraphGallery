import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import Planetoid

data = Planetoid('cora', root="~/GraphData/datasets/", verbose=False)

graph = data.graph
splits = data.split_nodes()
graph.update(node_attr=gf.normalize_attr(graph.node_attr))

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.DenseGCN(graph, device=device, seed=123).process().build(hids=32)
his = trainer.train(splits.train_nodes,
                    splits.val_nodes,
                    verbose=1,
                    epochs=100)

################### Attacker model ############################
attacker = gg.attack.untargeted.MinMax(graph, device=device, seed=None).process(trainer, splits.train_nodes,
                                                                                unlabeled_nodes=splits.test_nodes)
attacker.attack(0.05, CW_loss=True)

################### Victim model ############################
# This is a white-box attack
# Before attack
original_result = trainer.test(splits.test_nodes)

# After attack
trainer.graph = attacker.g
# reprocess after the graph has changed
trainer.process()  # important!
perturbed_result = trainer.test(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)

"""original prediction 82.70%
perturbed prediction 75.50%
The accuracy has gone down 7.20%"""
