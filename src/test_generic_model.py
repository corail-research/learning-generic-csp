import socket
from datetime import datetime
import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
from generic_xcsp.generic_model import GenericModel
from generic_xcsp.dataset import XCSP3Dataset
from generic_xcsp.training_config import GenericExperimentConfig
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataListLoader 
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# from torch_geometric.nn import DataParallel
from models.common.training_utils import train_model, process_model
from models.common.pytorch_samplers import  PairNodeSampler, PairBatchSampler, custom_batch_collate_fn

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
    import math
    import itertools
    

    search_method = "grid"  # Set to either "grid" or "random"
    # Hyperparameters for grid search or random search
    batch_sizes = 32
    hidden_units = 256
    num_lstm_passes = 26
    num_layers = 3
    device = "cuda"
    gnn_aggregation = "add"
    model_name = "fallen-gorge-29.pth"
    model_save_path = None
    # project_name = "Generic-SAT"
    project_name = "Generic-Knapsack"
    # project_name = "Generic-GC"
#    project_name = "Test"

    if project_name == "Generic-TSP":
        data_path = r"/scratch1/boileo/dtsp/data/generic_batched"
        model_save_path = r"/scratch1/boileo/dtsp/models"
    elif project_name == "Generic-TSP-Element":
        data_path = r"/scratch2/boileo/dtsp/data/generic_element_batched"
        model_save_path = r"/scratch1/boileo/dtsp/models"
    elif project_name == "Generic-GC":
        data_path = r"/scratch1/boileo/graph_coloring/data/generic"
        model_save_path = r"/scratch1/boileo/graph_coloring/models"
    elif project_name == "Generic-Knapsack":
        data_path = r"/scratch1/boileo/knapsack/test_data"
        model_save_path = r"/scratch1/boileo/knapsack/models"
    elif project_name == "Generic-SAT":
        data_path = r"/scratch1/boileo/sat/data/generic"
        model_save_path = r"/scratch1/boileo/sat/models"
    else:
        data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\data"
        model_save_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\models"
    
    hostname = socket.gethostname()

    model_path = os.path.join(model_save_path, model_name)
    dataset = XCSP3Dataset(root=data_path, in_memory=False, target_deviation=0.02)
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())
    
    # for params in search_parameters:
    loader = DataLoader(dataset, batch_size=32//32, shuffle=True, collate_fn=custom_batch_collate_fn, num_workers=0)
    
    first_batch = dataset[0][0]
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
    
    model = GenericModel(
        metadata=metadata,
        in_channels=input_size,
        out_channels=out_channels,
        hidden_size=hidden_size,
        num_mlp_layers=num_layers,
        num_passes=num_lstm_passes,
        device=device,
        layernorm_lstm_cell=True,
        aggr=gnn_aggregation
    )
    
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    
    group = hostname
    wandb.init(
        project=project_name,
        name=f"test_{model_name}",
        group=group
    )

    process_model(model, None, criterion, loader, mode='test')
    wandb.finish() 