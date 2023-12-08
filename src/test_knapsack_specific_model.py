import argparse
import socket
from datetime import datetime
import wandb
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from generic_xcsp.knapsack_model import KnapsackModel
from generic_xcsp.dataset import XCSP3Dataset

from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from models.common.training_utils import process_model
from models.common.pytorch_samplers import custom_batch_collate_fn

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create the parser
parser = argparse.ArgumentParser(description='Train a generic model')

# Add the arguments
parser.add_argument('--search_method', type=str, default='grid', help='Set to either "grid" or "random"')
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
parser.add_argument('--generic_representation', type=bool, default=True)
parser.add_argument('--gnn_aggregation', type=str, default='add')
parser.add_argument('--model_save_path', type=str, default=None)
parser.add_argument('--project_name', type=str, default='Knapsack')

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
gnn_aggregation = args.gnn_aggregation
model_save_path = args.model_save_path
layernorm_lstm_cell = True
project_name = args.project_name
print(batch_size)

if __name__ == "__main__":
    search_method = "grid"  # Set to either "grid" or "random"
    # Hyperparameters for grid search or random search
    model_save_path = None
    project_name = "Knapsack-Specific"

    data_path = r"/scratch1/boielo/knapsack/test_data"
    model_save_path = r"/scratch1/boielo/knapsack/models"
    model_name = "bright-shadow-22.pth"
    model_path = os.path.join(model_save_path, model_name)
    
    hostname = socket.gethostname()

    # Generate parameters based on the search method
    
    dataset = XCSP3Dataset(root=data_path, in_memory=False, batch_size=1, target_deviation=0.02, use_knapsack_specific_graph=True)
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())
    
    # train_loader = DataLoader(train_dataset, batch_size=32//32, shuffle=True, collate_fn=custom_batch_collate_fn, num_workers=0)
    loader = DataLoader(dataset, batch_size=4//4, collate_fn=custom_batch_collate_fn, num_workers=0)
    
    first_batch = dataset[0][0]
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
    
    
    model = KnapsackModel(
        metadata=metadata,
        in_channels=input_size,
        out_channels=out_channels,
        hidden_size=hidden_size,
        num_mlp_layers=num_layers,
        num_passes=num_lstm_passes,
        device=device,
        layernorm_lstm_cell=layernorm_lstm_cell,
        aggr=gnn_aggregation
    )
    
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    group = hostname
    wandb.init(
        project=project_name,
        name=f"test_{model_name}"
        group=group
    )
    
    
    process_model(model, None, criterion, loader, mode='test')
    
    wandb.finish()