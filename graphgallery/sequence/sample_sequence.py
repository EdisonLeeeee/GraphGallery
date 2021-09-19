
import numpy as np

from .sequence import Sequence


class SBVATSampleSequence(Sequence):

    def __init__(
        self,
        x,
        y,
        neighbors,
        out_index=None,
        sizes=50,
        resample=True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.out_index = out_index
        self.neighbors = neighbors
        self.num_nodes = x[0].shape[0]
        self.sizes = sizes
        self.adv_mask = self.smple_nodes()
        self.resample = resample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.astensors(*self.x, self.adv_mask), self.astensor(self.y), self.astensor(self.out_index)

    def on_epoch_end(self):
        if self.resample:
            self.adv_mask = self.smple_nodes()

    def smple_nodes(self):
        N = self.num_nodes
        flag = np.zeros(N, dtype=np.bool)
        adv_index = np.zeros(self.sizes, dtype='int32')
        for i in range(self.sizes):
            n = np.random.randint(0, N)
            while flag[n]:
                n = np.random.randint(0, N)
            adv_index[i] = n
            flag[self.neighbors[n]] = True
            if flag.sum() == N:
                break
        adv_mask = np.zeros(N, dtype='float32')
        adv_mask[adv_index] = 1.
        return adv_mask
