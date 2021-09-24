import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

################### Attacker model ############################
attacker = gg.attack.untargeted.RAND(graph, seed=123).process()
attacker.attack(0.05)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(seed=123).setup_graph(graph).build()
trainer.fit(splits.train_nodes,
            splits.val_nodes,
            verbose=1,
            epochs=100)
original_result = trainer.evaluate(splits.test_nodes)

# After attack
trainer = gg.gallery.nodeclas.GCN(seed=123).setup_graph(attacker.g).build()
trainer.fit(splits.train_nodes,
            splits.val_nodes,
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
perturbed prediction 82.39%
The accuracy has gone down 1.11%"""
