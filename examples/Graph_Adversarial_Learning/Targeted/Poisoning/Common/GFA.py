import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('citeseer',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

################### Attacker model ############################
target = 1
attacker = gg.attack.targeted.GFA(graph, seed=123).process()
attacker.attack(target)

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
"""original prediction [0.01540654 0.08327845 0.09508752 0.02212185 0.30293745 0.48116812]
perturbed prediction [0.00536365 0.07832371 0.5188041  0.0130601  0.17781597 0.2066326 ]
The True label of node 1 is 4.
The probability of prediction has gone down 0.12512147426605225"""
