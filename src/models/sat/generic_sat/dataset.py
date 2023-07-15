from typing import List
import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm
from sat_parser import parse_dimacs_cnf
import re
import os
import sys
import warnings
import random

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def _repr(obj) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

class SatDataset(InMemoryDataset):
    def __init__(self, root:str, transform:torch_geometric.transforms=None, pre_transform=None, graph_type:str="generic", 
                 meta_connected_to_all:bool=False, use_sat_label_as_feature:bool=False, in_memory: bool=True):
        """
        Args:
            root : directory containing the dataset. This directory contains 2 sub-directories: raw (raw data) and processed (processed data)
            transform (optional): transform to apply to elements of the dataset. Defaults to None.
            pre_transform (optional): Defaults to None.
            graph_type (str): choice of 'sat_specific', or 'generic'. Because the generic version needs one negation operator node per negated literal,
                it results in much more nodes in the graph. In addition to requiring less nodes to encode the problem, the sat_specific version connects positive literals
                to their negation, which gives more structure to the resulting graph.
            meta_connected_to_all (bool): If set to true, the meta node will be connected to all other nodes. Otherwise, only constraints will be
            use_sat_label_as_feature (bool): whether to use the label (sat or unsat) as a feature in the graph. Should be used for testing purposes only
        """
        self.graph_type = graph_type
        self.meta_connected_to_all = meta_connected_to_all
        self.use_sat_label_as_feature = use_sat_label_as_feature
        self.in_memory = in_memory
        super(SatDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        if self.in_memory:
            self.data = self.processed_data()
    
    @property
    def raw_file_names(self):
        """
        If this file exists in the raw_dir directory, files will not be dowloaded
        """
        return "sr_n=0006_pk2=0.30_pg=0.40_t=9_sat=0.dimacs"
    
    @property
    def processed_file_names(self):
        """If these files are present in the processed data directory, data processing step is skipped
        """
        # return "data_0_sat=0.pt"
        return "data"
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = os.listdir(self.raw_dir)

        return [os.path.join(self.raw_dir, f) for f in files]
    
    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = os.listdir(self.processed_dir)

        return [os.path.join(self.processed_dir, f) for f in files]

    def download(self):
        pass

    def process(self):
        sorted_raw_paths = sorted(self.raw_paths)
        num_files_to_process = len(self.raw_paths)
        
        pbar = tqdm(total=num_files_to_process, position=0)
        for i, filepath in enumerate(sorted_raw_paths):

            current_pair = i // 2
            current_element = i % 2
            pbar.update(1)
            cnf = parse_dimacs_cnf(filepath)
            
            if self.graph_type == "sat_specific":
                data = cnf.build_sat_specific_heterogeneous_graph(use_sat_label_as_feature=self.use_sat_label_as_feature)
            elif self.graph_type == "generic":
                data = cnf.build_generic_heterogeneous_graph(meta_connected_to_all=self.meta_connected_to_all)
            
            is_sat = filepath[-8]
            out_path = os.path.join(self.processed_dir, f"data_{current_pair}_{current_element}_sat={is_sat}.pt")
            torch.save(data, out_path, _use_new_zipfile_serialization=False)
        
        pbar.close()
    
    def _process(self):
        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")

        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.log:
            print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        if self.log:
            print('Done!', file=sys.stderr)
    
    def processed_data(self) -> List[torch.Tensor]:
        processed_paths = self.processed_paths
        return [torch.load(path) for path in processed_paths]
    
    def len(self):
        return len(self.processed_paths)
    
    def get(self, idx: int):
        path = self.processed_paths[idx]
        data = torch.load(path)
        
        return data