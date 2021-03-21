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
# Nettack takes no activation layer
trainer = gg.gallery.nodeclas.GCN(graph, seed=42).process().build(acts=None)
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
# surrogate weights
if gg.backend() == "tensorflow":
    w1, w2 = trainer.model.weights
    W = w1 @ w2
else:
    w1, w2 = trainer.model.parameters()
    W = (w2 @ w1).T
W = gf.tensoras(W)

################### Attacker model ############################
target = 0
attacker = gg.attack.targeted.Nettack(graph, seed=123).process(W)
attacker.attack(target,
                direct_attack=True,
                structure_attack=True,
                feature_attack=False)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(seed=123).make_data(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
original_predict = trainer.predict(target, transform="softmax")

# After attack
trainer = gg.gallery.nodeclas.GCN(seed=123).make_data(attacker.g).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
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
"""original prediction [0.00212943 0.0030072  0.90525377 0.03167017 0.01139321 0.00445553
 0.04209068]
perturbed prediction [5.7374395e-04 7.1133757e-03 2.0801996e-01 7.1272224e-01 1.5738427e-03
 9.5932820e-04 6.9037504e-02]
The True label of node 1 is 2.
The probability of prediction has gone down 0.6972337961196899"""
