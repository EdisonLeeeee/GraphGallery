import os
import zipfile
import os.path as osp
import numpy as np

from graphgallery.data import Dataset
from graphgallery.data.utils import makedirs, files_exist, download_file


class ZippedGraphDataset(Dataset):
    
    github_url = "https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/{}.zip"
    obj_names = ["adj.pkl", "feature.npy", "label.npy"]

    def __init__(self, name, root=None, url=None):
        super().__init__(name, root)
        
        self._url = url
    
        self.download_dir = osp.join(self.root, name, 'raw')
        self.processed_dir = osp.join(self.root, name, 'processed')
        
        makedirs(self.download_dir)
        makedirs(self.processed_dir)
        
        self.download()
        self.process()
    
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
        for raw_path in self.raw_paths:
            with zipfile.ZipFile(raw_path, 'r') as zipf:
                zipf.extractall(self.processed_dir)
        print("Process done.")

    
    @property
    def url(self):
        if isinstance(self._url, str):
            return self._url
        else:
            return self.github_url.format(self.name)
    
    @property
    def raw_file_names(self):
        return self.processed_obj_names
        
    @property
    def raw_paths(self):
        return [f"{osp.join(self.download_dir, self.name)}.zip"]
    
    @property
    def processed_obj_names(self):
        return self.obj_names
    
    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, fname) for fname in self.processed_obj_names]        