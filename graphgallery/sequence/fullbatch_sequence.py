from graphgallery import astensor, astensors
from graphgallery.sequence.node_sequence import NodeSequence


class FullBatchNodeSequence(NodeSequence):

    def __init__(self, inputs, labels=None):
        self.inputs = astensors(inputs)
        self.labels = astensor(labels)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.labels

    def on_epoch_end(self):
        pass

    def shuffle(self):
        pass
