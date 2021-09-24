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
trainer = gg.gallery.nodeclas.SGC(seed=123).setup_graph(graph, K=2).build()
trainer.fit(splits.train_nodes,
            splits.val_nodes,
            verbose=2,
            epochs=100)

################### Attacker model ############################
target = 1
W, b = trainer.model.weights
attacker = gg.attack.targeted.FasterSGA(graph, seed=123).process(W, b)
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
"""original prediction [0.0265792  0.0432898  0.8392475  0.05667492 0.01218924 0.00459338
 0.01742591]
perturbed prediction [0.00462345 0.7869288  0.14639236 0.05495493 0.00161338 0.00205874
 0.00342828]
The True label of node 1 is 2.
The probability of prediction has gone down 0.692855179309845"""
