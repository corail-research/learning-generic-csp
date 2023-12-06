from typing import List
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Batch
import torch_geometric
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
import re
import os
import sys


if __name__ == "__main__":
    from instance import parse_instance
    from graph_builder import XCSP3GraphBuilder
else:
    from generic_xcsp.instance import parse_instance
    from generic_xcsp.graph_builder import XCSP3GraphBuilder

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def _repr(obj) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

class XCSP3Dataset(Dataset):
    def __init__(self,
                 root:str,
                 transform:torch_geometric.transforms=None,
                 pre_transform=None,
                 target_deviation=0.02,
                 batch_size:int=32,
                 use_marty_et_al_graph:bool=False,
                 use_knapsack_specific_graph:bool=False,
                 in_memory: bool=True):
        """
        Args:
            root : directory containing the dataset. This directory contains 2 sub-directories: raw (raw data) and processed (processed data)
            transform (optional): transform to apply to elements of the dataset. Defaults to None.
            pre_transform (optional): Defaults to None.
        """
        self.in_memory = in_memory
        self.sorted_raw_paths = None
        self.sorted_processed_paths = None
        self.target_deviation = target_deviation
        self.batch_size = batch_size
        self.use_marty_et_al_graph = use_marty_et_al_graph
        self.use_knapsack_specific_graph = use_knapsack_specific_graph
        super(XCSP3Dataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        if self.in_memory:
            self.data = self.processed_data()
    
    @property
    def raw_file_names(self):
        """
        If this file exists in the raw_dir directory, files will not be dowloaded
        """
        return "0.xml"
    
    @property
    def processed_file_names(self):
        """If these files are present in the processed data directory, data processing step is skipped
        """
        # return /
        return False
    
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
        data_list = []
        pbar = tqdm(total=num_files_to_process, position=0)
        for current_pair, filepath in enumerate(self.raw_paths):
            pbar.update(1)
            xcsp3_instance = parse_instance(filepath, optimal_deviation_factor=self.target_deviation)
            graph_builder = XCSP3GraphBuilder(xcsp3_instance, filepath)
            if self.use_marty_et_al_graph:
                data = graph_builder.get_marty_et_al_graph_representation()
            elif self.use_knapsack_specific_graph:
                data = graph_builder.get_knapsack_specific_graph_representation()
            else:
                data = graph_builder.get_graph_representation()
            
            if type(data) == tuple:
                data_list.extend(data)
            else:
                data_list.append(data)
            if len(data_list) == self.batch_size:
                batch_id = current_pair // self.batch_size
                out_path = os.path.join(self.processed_dir, f'data_batch={batch_id}.pt')
                batched = Batch.from_data_list(data_list)
                torch.save(batched, out_path, _use_new_zipfile_serialization=False)
                data_list = []
        
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
    data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\data"
    dataset = XCSP3Dataset(root=data_path, use_knapsack_specific_graph=True, in_memory=False, target_deviation=None, batch_size=32)