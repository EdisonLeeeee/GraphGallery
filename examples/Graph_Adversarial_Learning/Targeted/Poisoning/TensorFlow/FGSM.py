import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

################### Surrogate model ############################
trainer = gg.gallery.DenseGCN(graph, seed=42).process().build()
his = trainer.train(splits.train_nodes,
                    splits.val_nodes,
                    verbose=1,
                    epochs=100)

################### Attacker model ############################
target = 1
attacker = gg.attack.targeted.FGSM(graph, seed=123).process(trainer)
attacker.attack(target)

################### Victim model ############################
# Before attack
trainer = gg.gallery.GCN(graph, seed=123).process().build()
his = trainer.train(splits.train_nodes,
                    splits.val_nodes,
                    verbose=1,
                    epochs=100)
original_predict = trainer.predict(target, return_logits=False)

# After attack
trainer = gg.gallery.GCN(attacker.g, seed=123).process().build()
his = trainer.train(splits.train_nodes,
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
perturbed prediction [0.00607361 0.8845625  0.0120528  0.02675715 0.00289692 0.02047921
 0.04717785]
The True label of node 1 is 2.
The probability of prediction has gone down 0.8932009935379028"""