try:
    import texttable
except ImportError:
    texttable = None
    
import numpy as np
import os.path as osp

from graphgallery.data.preprocess import train_val_test_split_tabular


class Dataset:
    def __init__(self, name, root=None, verbose=True):
        if root is None:
            root = 'dataset'
            
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))   
            
        root = osp.abspath(root)
        self.root = root
        self.name = name.lower()         
        self.verbose = verbose
        self.download_dir = None
        self.processed_dir = None
        self.graph = None
            
    @property
    def urls(self):
        return [self.url]
    
    @property
    def url(self):
        None
        
    def download(self):
        raise NotImplementedError
        
    def process(self):
        raise NotImplementedError
        
    def split(self, train_size=0.1, val_size=0.1, test_size=0.8,
              random_state=None):


        labels = self.graph.labels
        idx_train, idx_val, idx_test =  train_val_test_split_tabular(labels.shape[0], train_size, val_size, test_size,
                                                                     stratify=labels,
                                                                     random_state=random_state)
        
        return idx_train, idx_val, idx_test 
        
    @staticmethod   
    def print_files(file_paths):
        if not texttable:
            print(file_paths)
        else:
            t = texttable.Texttable()
            items = [(path.split('/')[-1], path) for path in file_paths]

            t.add_rows([['File Name', 'File Path'], *items])
            print(t.draw())    