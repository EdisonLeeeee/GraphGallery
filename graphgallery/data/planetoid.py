import os
import os.path as osp
import numpy as np
import pickle as pkl


from graphgallery.data import Dataset
from graphgallery.data.utils import makedirs, files_exist, download_file, process_planetoid_datasets


class Planetoid(Dataset):
    """The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    
    The original url is: <https://github.com/kimiyoung/planetoid/raw/master/data>
    """
    
    
    github_url = "https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/planetoid"
    obj_names = ["adj.pkl", "feature.npy", "label.npy", "idx_train.npy", "idx_val.npy", "idx_test.npy"]
    
    def __init__(self, name, root=None):
        super().__init__(name, root)
        
        self.download_dir = osp.join(self.root, 'planetoid', name, 'raw')
        self.processed_dir = osp.join(self.root, 'planetoid', name, 'processed')
        
        makedirs(self.download_dir)
        makedirs(self.processed_dir)
        
        self.download()
        self.process()
        
        self.adj, self.x, self.labels, self.idx_train, self.idx_val, self.idx_test = self.load()        
    
    def download(self):
        
        if files_exist(self.raw_paths):
            print(f"Downloaded dataset files exist in {self.raw_paths}")
            return 
        
        print("Downloading...")
        download_file(self.raw_paths, self.urls)
        print("Download done.")
        
    def process(self):
        
        if files_exist(self.processed_paths):
            print(f"Processed dataset files exist in {self.processed_paths}")
            return 
        
        print("Processing...")
        objs = process_planetoid_datasets(self.name, self.raw_paths)
        for fname, obj in zip(self.processed_paths, objs):
            if fname.endswith('pkl'):
                with open(fname, 'wb') as f:
                    pkl.dump(obj, f)
            elif fname.endswith('npy'):
                np.save(fname, obj)
            else:
                raise OSError(f"Unrecognized file name {fname}. Allowed file name `*.pkl` or `*.npy`.")               
        print("Process done.")

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
    
    @property
    def urls(self):
        return [f"{osp.join(self.github_url, raw_file_name)}" for raw_file_name in self.raw_file_names]

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]
    
    @property
    def raw_paths(self):
        return [f"{osp.join(self.download_dir, raw_file_name)}" for raw_file_name in self.raw_file_names]    
    
    @property
    def processed_obj_names(self):
        return self.obj_names
    
    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, fname) for fname in self.processed_obj_names]    
    
    
