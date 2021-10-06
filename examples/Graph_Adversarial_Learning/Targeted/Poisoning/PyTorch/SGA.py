import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

gg.set_backend("th")

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.SGC(seed=1000).setup_graph(graph, K=2).build(lr=0.01)
trainer.fit(splits.train_nodes,
            splits.val_nodes,
            verbose=2,
            epochs=100)

################### Attacker model ############################
target = 1
attacker = gg.attack.targeted.SGA(graph, seed=123).process(trainer)
attacker.attack(target)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(seed=123).setup_graph(graph).build()
trainer.fit(splits.train_nodes,
            splits.val_nodes,
            verbose=2,
            epochs=100)
original_predict = trainer.predict(target, transform="softmax")

# After attack
trainer = gg.gallery.nodeclas.GCN(seed=123).setup_graph(attacker.g).build()
trainer.fit(splits.train_nodes,
            splits.val_nodes,
            verbose=2,
            epochs=100)
perturbed_predict = trainer.predict(target, transform="softmax")

################### Results ############################
print("original prediction", original_predict)
print("perturbed prediction", perturbed_predict)
target_label = graph.node_label[target]
print(f"The True label of node {target} is {target_label}.")
print(
    f"The probability of prediction has gone down {original_predict[target_label]-perturbed_predict[target_label]}"
)
"""original prediction [0.0053769  0.0066478  0.9277275  0.02925558 0.02184986 0.00333208
 0.00581014]
perturbed prediction [0.00911093 0.00466003 0.32176667 0.63783115 0.00680091 0.0019819
 0.01784842]
The True label of node 1 is 2.
The probability of prediction has gone down 0.6059608459472656"""
