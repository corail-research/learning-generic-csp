from typing import List
import torch
from torch_geometric.data import InMemoryDataset, Dataset
import torch_geometric
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
try:
    from .gc_parser import GraphColoringInstance
except ImportError:
    from gc_parser import GraphColoringInstance

import re
import os
import sys

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def _repr(obj) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

class GraphColoringDataset(InMemoryDataset):
    def __init__(self, root:str, transform:torch_geometric.transforms=None, pre_transform=None, graph_type:str="generic", in_memory: bool=True):
        """
        Args:
            root : directory containing the dataset. This directory contains 2 sub-directories: raw (raw data) and processed (processed data)
            transform (optional): transform to apply to elements of the dataset. Defaults to None.
            pre_transform (optional): Defaults to None.
            graph_type (str): choice of 'gc_specific', or 'generic'.
        """
        self.graph_type = graph_type
        self.in_memory = in_memory
        self.sorted_raw_paths = None
        self.sorted_processed_paths = None
        self.color_embeddings = torch.rand(10, 1)
        super(GraphColoringDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        if self.in_memory:
            self.data = self.processed_data()
    
    @property
    def raw_file_names(self):
        """
        If this file exists in the raw_dir directory, files will not be dowloaded
        """
        return "0.graph"
    
    @property
    def processed_file_names(self):
        """If these files are present in the processed data directory, data processing step is skipped
        """
        return "data"
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        if self.sorted_raw_paths is None:
            files = os.listdir(self.raw_dir)
            self.sorted_raw_paths = sorted([os.path.join(self.raw_dir, f) for f in files])
        return self.sorted_raw_paths
    
    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        if self.sorted_processed_paths is None:
            files = os.listdir(self.processed_dir)
            self.sorted_processed_paths = sorted([os.path.join(self.processed_dir, f) for f in files])

        return self.sorted_processed_paths

    def process(self):
        if self.sorted_raw_paths is None:
            self.raw_paths
        num_files_to_process = len(self.raw_paths)
        num_digits = len(str(num_files_to_process)) # zero pad the file names so that they are sorted correctly
        pbar = tqdm(total=num_files_to_process, position=0)
        
        for pair_id, filepath in enumerate(self.raw_paths):
            if self.graph_type == "gc_specific":
                raise NotImplementedError
            elif self.graph_type == "generic":
                data_positive, data_negative = GraphColoringInstance(filepath, self.color_embeddings).get_gc_specific_representation()
            
            positive_out_path = os.path.join(self.processed_dir, f"data_{str(pair_id).zfill(num_digits)}_1.pt")
            negative_out_path = os.path.join(self.processed_dir, f"data_{str(pair_id).zfill(num_digits)}_0.pt")
            torch.save(data_positive, positive_out_path, _use_new_zipfile_serialization=False)
            torch.save(data_negative, negative_out_path, _use_new_zipfile_serialization=False)
        
        pbar.close()
    
    def _process(self):
        if len(os.listdir(self.processed_dir)) != 0:
            return
        if self.log:
            print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        if self.log:
            print('Done!', file=sys.stderr)
    
    def __getitems__(self, idx):
        if isinstance(idx[0], list):
            return [self[i] for i in idx[0]]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, list):
            return [self[i] for i in idx]
        else:
            raise TypeError('Invalid argument type')
    
    def processed_data(self) -> List[torch.Tensor]:
        processed_paths = self.processed_paths
        return [torch.load(path) for path in processed_paths]
    
    def len(self):
        return len(self.processed_paths)
    
    def get(self, idx: int):
        path = self.processed_paths[idx]
        data = torch.load(path)
        
        return data

if __name__ == "__main__":
    data_path = "data"
    dataset = GraphColoringDataset(data_path, graph_type="generic")
    a = 1