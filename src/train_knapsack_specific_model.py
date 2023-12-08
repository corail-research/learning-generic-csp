import argparse
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
from generic_xcsp.knapsack_model import KnapsackModel
from generic_xcsp.dataset import XCSP3Dataset
from generic_xcsp.training_config import GenericTrainingConfig

from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from models.common.training_utils import train_model
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

# Create the parser
parser = argparse.ArgumentParser(description='Train a generic model')

# Add the arguments
parser.add_argument('--search_method', type=str, default='grid', help='Set to either "grid" or "random"')
parser.add_argument('--batch_size', type=int, default=2)
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
project_name = args.project_name
print(batch_size)

if __name__ == "__main__":
    search_method = "grid"  # Set to either "grid" or "random"
    # Hyperparameters for grid search or random search
    model_save_path = None
    project_name = "Knapsack-Specific"

    data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\data"
    model_save_path = None# r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\models"
    
    hostname = socket.gethostname()

    experiment_config = GenericTrainingConfig(
        batch_size=batch_size,
        hidden_units=hidden_units,
        start_learning_rate=start_learning_rate,
        num_layers=num_layers,
        dropout=dropout,
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
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_decay_factor,
        layernorm_lstm_cell=True,
        gnn_aggregation=gnn_aggregation,
        model_save_path=model_save_path,
        optimal_deviation_difference=0.02
    )

    # Generate parameters based on the search method
    
    dataset = XCSP3Dataset(root=experiment_config.data_path, in_memory=False, batch_size=4, target_deviation=0.02, use_knapsack_specific_graph=True)
    limit_index = int(((len(dataset) * experiment_config.train_ratio) // 2) * 2)
    
    train_dataset = dataset[:limit_index]
    test_dataset = dataset[limit_index:]
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())
    
    # train_loader = DataLoader(train_dataset, batch_size=32//32, shuffle=True, collate_fn=custom_batch_collate_fn, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=4//4, collate_fn=custom_batch_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=min(1, len(test_dataset)), shuffle=False, collate_fn=custom_batch_collate_fn, num_workers=0)
    
    first_batch = train_dataset[0][0]
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = experiment_config.hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
    
    
    model = KnapsackModel(
        metadata=metadata,
        in_channels=input_size,
        out_channels=out_channels,
        hidden_size=hidden_size,
        num_mlp_layers=experiment_config.num_layers,
        num_passes=experiment_config.num_lstm_passes,
        device=device,
        layernorm_lstm_cell=experiment_config.layernorm_lstm_cell,
        aggr=experiment_config.gnn_aggregation
    )

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=experiment_config.start_learning_rate, weight_decay=experiment_config.weight_decay)
    group = hostname
    wandb.init(
        project=project_name,
        config=experiment_config,
        group=group
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=experiment_config.lr_scheduler_patience, gamma=experiment_config.lr_scheduler_factor, verbose=True)
    
    train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        criterion,
        experiment_config.num_epochs,
        samples_per_epoch=experiment_config.samples_per_epoch,
        clip_value=experiment_config.clip_gradient_norm,
        model_save_path=experiment_config.model_save_path,
        wandb_run_name=wandb.run.name
    )
    
    wandb.finish()