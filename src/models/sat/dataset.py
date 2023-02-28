from typing import List
import torch
from torch_geometric.data import Dataset
import torch_geometric
from tqdm import tqdm
from sat_parser import parse_dimacs_cnf
import os


class SatDataset(Dataset):
    def __init__(self, root:str, transform:torch_geometric.transforms=None, pre_transform=None):
        """
        Args:
            root : directory containing the dataset. This directory contains 2 sub-directories: raw (raw data) and processed (processed data)
            transform (optional): transform to apply to elements of the dataset. Defaults to None.
            pre_transform (optional): Defaults to None.
        """
        super(SatDataset, self).__init__(root, transform, pre_transform)
    
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
        return "data_0_sat=0.pt"
    
    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = os.listdir(self.raw_dir)

        return [os.path.join(self.raw_dir, f) for f in files]

    def download(self):
        pass

    def process(self):
        num_files_to_process = len(self.raw_paths)
        pbar = tqdm(total=num_files_to_process, position=0)
        for i, filepath in enumerate(self.raw_paths):
            pbar.update(1)
            cnf = parse_dimacs_cnf(filepath)
            data = cnf.build_heterogeneous_graph()
            is_sat = filepath[-8]
            out_path = os.path.join(self.processed_dir, f"data_{i}_sat={is_sat}.pt")
            torch.save(data, out_path, _use_new_zipfile_serialization=False)
        
        pbar.close()

if __name__ == "__main__":
    dataset = SatDataset(root=r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic Graph Representation\Graph-Representation\src\models\sat\data")
    a=1