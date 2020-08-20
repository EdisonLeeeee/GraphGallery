import numpy as np
import os.path as osp


class Dataset:
    def __init__(self, name, root=None):
        if root is None:
            root = 'data'
            
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))   
            
        root = osp.abspath(root)
        
        self.root = root
        self.name = name           
        self.download_dir = None
        self.processed_dir = None
            
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
        

    def load(self):
        
        objs = []
        for fname in self.processed_paths:
            if fname.endswith('pkl') or fname.endswith('npy'):
                with open(fname, 'rb') as f:
                    obj = np.load(f, allow_pickle=True)
            else:
                raise OSError(f"Unrecognized file name {fname}. Allowed file name `*.pkl` or `*.npy`.")
                
            objs.append(obj)
            
        return objs