import socket
from datetime import datetime
import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from generic_xcsp.dataset import XCSP3Dataset
from models.sat.dataset import SatDataset

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

if __name__ == "__main__":

    # root_data_path = r"/scratch2/boileo/marty_et_al_data/"
    root_data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\test_marty_et_al_graphs"
    
    
    sat_data_path = os.path.join(root_data_path, "sat")
    dataset = SatDataset(root=sat_data_path, use_marty_et_al_graph=True, in_memory=False)
    graph_coloring_data_path = os.path.join(root_data_path, "graph_coloring")
    dataset = XCSP3Dataset(root=graph_coloring_data_path, in_memory=False, target_deviation=None, batch_size=32, use_marty_et_al_graph=True)
    dtsp_data_path = os.path.join(root_data_path, "dtsp")
    dataset = XCSP3Dataset(root=dtsp_data_path, in_memory=False, target_deviation=0.02, batch_size=32, use_marty_et_al_graph=True)
    knapsack_data_path = os.path.join(root_data_path, "knapsack")
    dataset = XCSP3Dataset(root=knapsack_data_path, in_memory=False, target_deviation=0.02, batch_size=32, use_marty_et_al_graph=True)