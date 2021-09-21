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
trainer = gg.gallery.nodeclas.DenseGCN(device=device, seed=123).setup_graph(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)


################### Attacker model ############################
victim_nodes = np.hstack([splits.val_nodes, splits.test_nodes])
victim_labels = trainer.predict(victim_nodes).argmax(1)

attacker = gg.attack.untargeted.FGSM(graph, device=device, seed=42).process(trainer, victim_nodes, victim_labels=victim_labels)
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
# reprocess after the graph has changed
trainer.setup_graph(attacker.g)  # important!
perturbed_result = trainer.evaluate(splits.test_nodes)

################### Results ############################
print(f"original prediction {original_result.accuracy:.2%}")
print(f"perturbed prediction {perturbed_result.accuracy:.2%}")
print(
    f"The accuracy has gone down {original_result.accuracy-perturbed_result.accuracy:.2%}"
)
"""original prediction 83.50%
perturbed prediction 79.83%
The accuracy has gone down 3.67%
"""
