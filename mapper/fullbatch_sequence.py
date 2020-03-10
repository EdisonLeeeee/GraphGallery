from .node_sequence import NodeSequence

class FullBatchNodeSequence(NodeSequence):

    def __init__(self, inputs, labels=None):

        self.inputs = self._to_tensor(inputs)
        self.labels = self._to_tensor(labels)
        self.n_batches = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.labels
    
    def on_epoch_end(self):
        pass
    
    def shuffle(self):
        pass