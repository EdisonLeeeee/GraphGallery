import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

# use PyTorch backend
gg.set_backend("torch")

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.DenseGCN(device=device, seed=123).setup_graph(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)

################### Attacker model ############################
attacker = gg.attack.untargeted.PGD(graph, device=device, seed=123).process(trainer, splits.train_nodes)
attacker.attack(0.05, CW_loss=False)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(device=device, seed=123).setup_graph(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
original_result = trainer.evaluate(splits.test_nodes)

# After attack
# If a validation set is used, the attacker will be less effective, but we dont know why
trainer = gg.gallery.nodeclas.GCN(attacker.g, device=device, seed=123).process().build()
his = trainer.fit(splits.train_nodes,
                  #                     splits.val_nodes,
                  verbose=1,
                  epochs=100)
perturbed_result = trainer.evaluate(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)
"""original prediction 85.06%
perturbed prediction 77.31%
The accuracy has gone down 7.75%"""
