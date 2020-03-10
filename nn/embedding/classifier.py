import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from . import DeepWalk, Node2Vec

class Classifier:

    def __init__(self, adj, labels, name='deepwalk'):
        assert name in ('deepwalk', 'node2vec')
        self.adj = adj
        self.labels = labels
        self.n_classes = labels.max() + 1        
        self.name = name
        self.embeddings = None
        self.embedding_model = None
        self.clf = None

    def build(self, solver='lbfgs', max_iter=1000):
        self.clf = LogisticRegression(solver=solver, max_iter=max_iter, multi_class='auto')

    def get_embeddings(self):
        
        if self.name == 'deepwalk':
            self.embedding_model = DeepWalk(self.adj)
            self.embedding_model.train()
            
        elif self.name == 'node2vec':
            self.embedding_model = Node2Vec(self.adj)
            self.embedding_model.train()
            
        else:
            raise ValueError(f'Invalid argument for name: {self.name}')   
            
            
        return self.embedding_model.get_embeddings()
            

    def train(self, index):
        if self.embeddings is None:
            self.embeddings = self.get_embeddings()
            
        self.clf.fit(self.embeddings[index], self.labels[index])

    def predict(self, index):
        logit = self.clf.predict_proba(self.embeddings[index])
        return logit
    
    def test(self, index):
        y_true =  self.labels[index]
        y_pred = self.clf.predict(self.embeddings[index])
        acc = accuracy_score(y_true, y_pred)
        return acc
        


