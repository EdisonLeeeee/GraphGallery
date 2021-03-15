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
target = 1
attacker = gg.attack.targeted.RAND(graph, seed=123).process()
attacker.attack(target)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(graph, seed=123).process().build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
original_predict = trainer.predict(target, return_logits=False)

# After attack
trainer = gg.gallery.nodeclas.GCN(attacker.g, seed=123).process().build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
perturbed_predict = trainer.predict(target, return_logits=False)

################### Results ############################
print("original prediction", original_predict)
print("perturbed prediction", perturbed_predict)
target_label = graph.node_label[target]
print(f"The True label of node {target} is {target_label}.")
print(
    f"The probability of prediction has gone down {original_predict[target_label]-perturbed_predict[target_label]}"
)
"""original prediction [0.00212943 0.0030072  0.90525377 0.03167017 0.01139321 0.00445553
 0.04209068]
perturbed prediction [0.00708749 0.01280669 0.74210054 0.11993515 0.01809976 0.00931118
 0.09065934]
The True label of node 1 is 2.
The probability of prediction has gone down 0.16315323114395142"""
