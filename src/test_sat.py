import socket
from datetime import datetime
import wandb
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from models.common.lstm_conv import AdaptedNeuroSAT
from models.sat.neurosat_model import NeuroSAT
from models.sat.dataset import SatDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader as BaseDataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import process_model
from models.sat.config import SATExperimentConfig
from models.common.pytorch_samplers import  PairNodeSampler, PairBatchSampler, custom_hetero_collate_fn

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    import math
    search_method = "grid"  # Set to either "grid" or "random"
    
    # Hyperparameters for grid search or random search
    pre_batched_data = True
    hidden_units = 128
    num_lstm_passes = 26
    num_layers = 3
    device = "cuda:0"
    model_name = "denim-salad-235.pth"
    model_directory = "/scratch1/boileo/sat/models"
    model_path = os.path.join(model_directory, model_name)

    generic_representation = False
    gnn_aggregation = "add"

    if generic_representation:
        flip_inputs = False
        data_path = r"/scratch1/boileo/sat/data/generic"
    else:
        flip_inputs = True
        data_path = r"/scratch1/boileo/sat/data/sat_specific"
    
    set_seed(42)

    hostname = socket.gethostname()

    # Generate parameters based on the search method
   
    if generic_representation:
        dataset = SatDataset(root=data_path, graph_type="generic", meta_connected_to_all=False, in_memory=False)
    else:
        dataset = SatDataset(root=data_path, graph_type="sat_specific", meta_connected_to_all=False, in_memory=False)
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())
    
    if pre_batched_data:
        loader = BaseDataLoader(dataset, batch_size=1024//32, shuffle=False, collate_fn=custom_hetero_collate_fn, num_workers=0)
        first_batch = dataset[0][0]
    else:
        loader = GeometricDataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        first_batch = dataset[0]
    
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}

    model = NeuroSAT(
        metadata,
        input_size,
        out_channels,
        hidden_size,
        num_passes=num_lstm_passes,
        device=device,
        flip_inputs=flip_inputs,
        layernorm_lstm_cell=True,
        aggr=gnn_aggregation,
        num_mlp_layers=num_layers
    )
    
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    
    if type(model) == AdaptedNeuroSAT:
        group = "generic" + hostname
    else:
        group = "sat_specific" + hostname
    wandb.init(
        project=f"SATGNN",
        group=group,
        name=f"test_{model_name}"
    )
    process_model(model, None, criterion, loader, mode='test')

    wandb.finish() 