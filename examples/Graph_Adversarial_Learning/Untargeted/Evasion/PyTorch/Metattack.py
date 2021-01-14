import numpy as np
import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora', root="~/GraphData/datasets/", verbose=False, transform="standardize")
graph = data.graph
splits = data.split_nodes(random_state=15)

# use PyTorch backend
gg.set_backend("torch")

# GPU is recommended
device = "gpu"

################### Surrogate model ############################
trainer = gg.gallery.GCN(graph, device=device, seed=None).process().build()
his = trainer.train(splits.train_nodes,
                    splits.val_nodes,
                    verbose=1,
                    epochs=200)

################### Attacker model ############################
unlabeled_nodes = np.hstack([splits.val_nodes, splits.test_nodes])
self_training_labels = trainer.predict(unlabeled_nodes).argmax(1)

attacker = gg.attack.untargeted.Metattack(graph, device=device, seed=123).process(splits.train_nodes,
                                                                                  unlabeled_nodes,
                                                                                  self_training_labels,
                                                                                  lr=0.1, # cora lr=0.1, citeseer lr=0.01 reaches the best performance
                                                                                  lambda_=0.,
                                                                                  use_relu=False) 
attacker.attack(0.05)
################### Victim model ############################
# This is a white-box attack
# Before attack
original_result = trainer.test(splits.test_nodes)

# After attack
trainer.graph = attacker.g
# reprocess after the graph has changed
trainer.process() # important!
perturbed_result = trainer.test(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)
