import argparse
import socket
from datetime import datetime
import wandb
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from models.common.lstm_conv import AdaptedNeuroSAT
from models.sat.neurosat_model import NeuroSAT
from models.sat.dataset import SatDataset
from torch.utils.data import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import train_model
from models.sat.config import SATTrainingConfig
from src.models.common.pytorch_utilities import custom_hetero_list_collate_fn

import random
import numpy as np
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create the parser
parser = argparse.ArgumentParser(description='Train a SAT model')

# Add the arguments
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_units', type=int, default=128)
parser.add_argument('--start_learning_rate', type=float, default=0.00002)
parser.add_argument('--num_lstm_passes', type=int, default=30)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_ratio', type=float, default=0.99)
parser.add_argument('--samples_per_epoch', type=int, default=100000)
parser.add_argument('--nodes_per_batch', type=int, default=12000)
parser.add_argument('--use_sampler_loader', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=0.0000000001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_scheduler_patience', type=float, default=25)
parser.add_argument('--clip_gradient_norm', type=float, default=0.5)
parser.add_argument('--generic_representation', type=bool, default=True)
parser.add_argument('--gnn_aggregation', type=str, default='add')
parser.add_argument('--model_save_path', type=str, default=None)
parser.add_argument('--project_name', type=str, default='SAT')
parser.add_argument('--data_path', type=str)
parser.add_argument('--model_save_path', type=str)

# Parse the arguments
args = parser.parse_args()

batch_size = args.batch_size
hidden_units = args.hidden_units
start_learning_rate = args.start_learning_rate
num_lstm_passes = args.num_lstm_passes
num_layers = args.num_layers
dropout = args.dropout
num_epochs = args.num_epochs
device = args.device
train_ratio = args.train_ratio
samples_per_epoch = args.samples_per_epoch
nodes_per_batch = args.nodes_per_batch
use_sampler_loader = args.use_sampler_loader
weight_decay = args.weight_decay
lr_decay_factor = args.lr_decay_factor
lr_scheduler_patience = args.lr_scheduler_patience
clip_gradient_norm = args.clip_gradient_norm
gnn_aggregation = args.gnn_aggregation
model_save_path = args.model_save_path
project_name = args.project_name
data_path = args.data_path
model_save_path = args.model_save_path
generic_representation = args.generic_representation


if __name__ == "__main__":
    hostname = socket.gethostname()

    if generic_representation:
        flip_inputs = False
    else:
        flip_inputs = True

    experiment_config = SATTrainingConfig(
        batch_sizes=batch_size,
        hidden_units=hidden_units,
        start_learning_rates=start_learning_rate,
        num_layers=num_layers,
        dropouts=dropout,
        num_epochs=num_epochs,
        num_lstm_passes=num_lstm_passes,
        device=device,
        train_ratio=train_ratio,
        samples_per_epoch=samples_per_epoch,
        nodes_per_batch=nodes_per_batch,
        data_path=data_path,
        use_sampler_loader=use_sampler_loader,
        weight_decay=weight_decay,
        lr_decay_factor=lr_decay_factor,
        generic_representation=generic_representation,
        flip_inputs=flip_inputs,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_decay_factor,
        layernorm_lstm_cell=True,
        gnn_aggregation=gnn_aggregation,
        model_save_path=model_save_path
    )


    if generic_representation:
        dataset = SatDataset(root=data_path, graph_type="generic", meta_connected_to_all=False, in_memory=False)
    else:
        dataset = SatDataset(root=data_path, graph_type="sat_specific", meta_connected_to_all=False, in_memory=False)
    
    train_dataset = dataset[:math.floor(len(dataset) * train_ratio)]
    test_dataset = dataset[math.floor(len(dataset) * train_ratio):]
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    # for params in search_parameters:
    train_loader = DataLoader(train_dataset, batch_size=batch_size//32, shuffle=True, collate_fn=custom_hetero_list_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024//32, shuffle=False, collate_fn=custom_hetero_list_collate_fn, num_workers=0)
    first_batch = train_dataset[0][0]
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
        aggr=gnn_aggregation
    )
    # model = AdaptedNeuroSAT(metadata, input_size, out_channels, hidden_size, num_passes=num_lstm_passes, device=device)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=start_learning_rate, weight_decay=weight_decay)
    if type(model) == AdaptedNeuroSAT:
        group = "generic" + hostname
    else:
        group = "sat_specific" + hostname
    wandb.init(
        project=f"SATGNN",
        config=experiment_config,
        group=group
    )
    if lr_scheduler_patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_patience, gamma=lr_decay_factor, verbose=True)
    else:
        lr_scheduler = None
    train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        criterion,
        num_epochs,
        samples_per_epoch=samples_per_epoch,
        clip_value=clip_gradient_norm,
        model_save_path=model_save_path,
        wandb_run_name=wandb.run.name
    )
    
    wandb.finish() 