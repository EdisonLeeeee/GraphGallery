from graphgallery.sequence.base_sequence import Sequence


class FullBatchNodeSequence(Sequence):

    def __init__(self, x, y=None, *args, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            x: (int): write your description
            y: (int): write your description
        """
        super().__init__(*args, **kwargs)

        self.x = self.astensors(x, device=self.device)
        self.y = self.astensor(y, device=self.device)

    def __len__(self):
        """
        Returns the number of rows

        Args:
            self: (todo): write your description
        """
        return 1

    def __getitem__(self, index):
        """
        Return the item at index

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        return self.x, self.y

    def on_epoch_end(self):
        """
        On end of epoch.

        Args:
            self: (todo): write your description
        """
        ...

    def _shuffle_batches(self):
        """
        Shuffle the batches of all the batches.

        Args:
            self: (todo): write your description
        """
        ...
