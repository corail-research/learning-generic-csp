import argparse
import socket
from datetime import datetime
import wandb
import math
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from models.graph_coloring.base_model import GCGNN
from models.graph_coloring.dataset import GraphColoringDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import train_model
from models.graph_coloring.config import GraphColoringConfig
from src.models.common.pytorch_utilities import PairNodeSampler, PairBatchSampler

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

    experiment_config = GraphColoringConfig(
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
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_decay_factor,
        layernorm_lstm_cell=True,
        gnn_aggregation=gnn_aggregation,
        clip_gradient_norm=clip_gradient_norm
    )

    if experiment_config.generic_representation:
        dataset = GraphColoringDataset(root=experiment_config.data_path, graph_type="generic")
    else:
        dataset = GraphColoringDataset(root=experiment_config.data_path, graph_type="gc_specific")
    delimiter_index = math.floor(len(dataset) * (experiment_config.train_ratio))
    delimiter_index -= delimiter_index % 2 # round to the closest even number
    train_dataset = dataset[:delimiter_index]
    test_dataset = dataset[delimiter_index:]
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())


    PairBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=0)
    first_batch_iter = iter(train_loader)
    first_batch = next(first_batch_iter)
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
    
    model = GCGNN(metadata, input_size, out_channels, hidden_size, num_passes=num_lstm_passes, device=device)
    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=start_learning_rate, weight_decay=weight_decay)
    if lr_scheduler_patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=lr_scheduler_patience, verbose=True)
    else:
        lr_scheduler = None

    if type(model) == GCGNN:
        group = "gc_specific"
    else:
        group = "generic"
    wandb.init(
        project=f"GC-GNN",
        config=experiment_config,
        group=group
    )
    train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        criterion,
        num_epochs,
        samples_per_epoch=samples_per_epoch,
        clip_value=experiment_config.clip_gradient_norm,
        model_save_path=model_save_path,
        wandb_run_name=wandb.run.name
    )
    
    wandb.finish() 