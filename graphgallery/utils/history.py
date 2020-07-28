try:
    import texttable
except ImportError:
    texttable = None
import numpy as np
from collections import defaultdict


class History:

    def __init__(self, monitor_metric='val_acc', early_stop_metric='val_loss'):

        self._check_name(monitor_metric)
        self._check_name(early_stop_metric)
        self.monitor_metric = monitor_metric
        self.early_stop_metric = early_stop_metric

        self._history = None
        self.epoch = None
        self.times = None
        self.patience = None
        self.best_results = None
        self.save_best = None
        self.early_stop_at = None

        self.reset()

    def reset(self):
        self._history = defaultdict(list)
        self.epoch = []
        self.times = 0
        self.patience = {'loss': 0, 'val_loss': 0,
                         'acc': 0, 'val_acc': 0}
        self.best_results = {'loss': np.inf, 'val_loss': np.inf,
                             'acc': -np.inf, 'val_acc': -np.inf}
        self.save_best = False
        self.early_stop_at = None

    def register_monitor_metric(self, name):
        self._check_name(name)
        self.monitor_metric = name

    def register_early_stop_metric(self, name):
        self._check_name(name)
        self.early_stop_metric = name

    def add_results(self, value, name):
        self._check_name(name)
        self.save_best_func(value, name)
        self.early_stop_func(value, name)

        self._history[name].append(value)

    def save_best_func(self, value, name):

        self.save_best = False

        if (name.endswith('acc') and value > self.best_results[name] or
                name.endswith('loss') and value < self.best_results[name]):
            self.best_results[name] = value

            if name == self.monitor_metric:
                self.save_best = True

    def early_stop_func(self, value, name):

        if self.times == 0:
            return

        if (name.endswith('acc') and value < self._history[name][-1] or
                name.endswith('loss') and value > self._history[name][-1]):
            self.patience[name] -= 1
        else:
            self.patience[name] = 0

#     def latest_results(self, n, name='val_loss', method='mean'):
#         Method = {'mean': np.mean, 'max': np.max, 'min': np.min, 'sum': np.sum}[method]
#         return Method(self._history[name][-(n+1):-1])

    def record_epoch(self, epoch):
        self.epoch.append(epoch)
        self.times += 1

    def time_to_early_stopping(self, early_stopping_patience):

        name = self.early_stop_metric
        if self.patience[name] + early_stopping_patience <= 0:
            self.early_stop_at = self.epoch[-1]
            return True
        else:
            return False

    @staticmethod
    def _check_name(name):
        assert name in ('loss', 'val_loss', 'acc', 'val_acc')

    @property
    def history(self):
        return dict(self._history)

    def show(self):
        assert texttable, "Please install `texttable` package!"
        t = texttable.Texttable()
        t.add_rows([["Training Details", "Value"],
                    ['Running times', self.times],
                    ['Lowerst training loss', self.best_results['loss']],
                    ['Highest training accuracy', self.best_results['acc']],
                    ['Lowerst validation loss', self.best_results['val_loss']],
                    ['Highest validation accuracy', self.best_results['val_acc']],
                    ['Metric to save best weights', self.monitor_metric],
                    ['Metric to early stopping', self.early_stop_metric],
                    ['Early stopping at', f'Epoch: {self.early_stop_at}']])

        print(t.draw())
