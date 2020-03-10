import numpy as np
import texttable
from collections import defaultdict


class History:
    
    def __init__(self, best_metric='val_accuracy', early_stop_metric='val_loss'):
        
        self._check_name(best_metric)
        self._check_name(early_stop_metric)
        self.best_metric = best_metric
        self.early_stop_metric = early_stop_metric
                
        self.history = None
        self.epoch = None
        self.times = None
        self.patience = None
        self.best_results = None
        self.restore_best = None
        self.early_stop_at = None
        
        self.reset()
        
    def reset(self):
        self.history = defaultdict(list)
        self.epoch = []
        self.times = 0
        self.patience = {'loss':0, 'val_loss':0, 
                         'accuracy': 0, 'val_accuracy': 0}
        self.best_results = {'loss': np.inf, 'val_loss': np.inf, 
                             'accuracy': -np.inf, 'val_accuracy': -np.inf}
        self.restore_best = False
        self.early_stop_at = None
        
    def register_best_metric(self, name):
        self._check_name(name)
        self.best_metric = name
        
    def register_early_stop_metric(self, name):
        self._check_name(name)
        self.early_stop_metric = name      
        
    def add_results(self, value, name):
        self._check_name(name)
        self.restore_best_func(value, name)
        self.early_stop_func(value, name)
            
        self.history[name].append(value)
        
    def restore_best_func(self, value, name):
        
        self.restore_best = False
        
        if (name.endswith('accuracy') and value > self.best_results[name] or
            name.endswith('loss') and value < self.best_results[name]):
            self.best_results[name] = value

            if name == self.best_metric:
                self.restore_best = True
    
    def early_stop_func(self, value, name):
        
        if self.times == 0:
            return 
        
        if (name.endswith('accuracy') and value < self.history[name][-1] or
            name.endswith('loss') and value > self.history[name][-1]):
            self.patience[name] -= 1
        else:
            self.patience[name] = 0
        
#     def latest_results(self, n, name='val_loss', method='mean'):
#         Method = {'mean': np.mean, 'max': np.max, 'min': np.min, 'sum': np.sum}[method]
#         return Method(self.history[name][-(n+1):-1])
    
    def record_epoch(self, epoch):
        self.epoch.append(epoch)
        self.times += 1
        
    def time_to_early_stopping(self, early_stopping_patience):
        
        name = self.early_stop_metric
        if self.patience[name] + early_stopping_patience <=0:
            self.early_stop_at = self.epoch[-1]
            return True
        else:
            return False
        
    @staticmethod    
    def _check_name(name):
        assert name in ('loss', 'val_loss', 'accuracy', 'val_accuracy')
        
    def show(self):
        t = texttable.Texttable()
        t.add_rows([["Training Details", "Value"],
                    ['Running times', self.times],
                    ['Lowerst training loss', self.best_results['loss']],
                    ['Highest training accuracy', self.best_results['accuracy']],
                    ['Lowerst validation loss', self.best_results['val_loss']],
                    ['Highest validation accuracy', self.best_results['val_accuracy']],
                    ['Metric to restore best weights', self.best_metric],
                    ['Metric to early stopping', self.early_stop_metric],
                    ['Early stopping at', f'Epoch: {self.early_stop_at}']])
        
        print(t.draw())  