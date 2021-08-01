import numpy as np
import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.GCN(device=device, seed=None).setup_graph(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=200)

################### Attacker model ############################
unlabeled_nodes = np.hstack([splits.val_nodes, splits.test_nodes])
self_training_labels = trainer.predict(unlabeled_nodes).argmax(1)

attacker = gg.attack.untargeted.MetaApprox(graph, device=device, seed=123).process(splits.train_nodes,
                                                                                   unlabeled_nodes,
                                                                                   self_training_labels,
                                                                                   lr=0.1,  # cora lr=0.1, citeseer lr=0.01 reaches the best performance
                                                                                   lambda_=.5,
                                                                                   use_relu=False)
attacker.attack(0.05)

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
"""original prediction 83.50%
perturbed prediction 79.38%
The accuracy has gone down 4.12%"""
