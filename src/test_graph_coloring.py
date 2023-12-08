import socket
from datetime import datetime
import wandb
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from models.graph_coloring.base_model import GCGNN
from models.graph_coloring.dataset import GraphColoringDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import process_model

if __name__ == "__main__":
    search_method = "random"  # Set to either "grid" or "random"
    data_path = r"/scratch1/boileo/graph_coloring/data/gc_specific"
    
    # Hyperparameters for grid search or random search
    batch_size = 128
    hidden_units = 64
    num_lstm_passes = 26
    num_layers = 3
    device = "cuda:0"
    generic_representation = False
    gnn_aggregation = "add"
    model_name = "denim-salad-235.pth"
    model_save_path = "/scratch1/boileo/graph_coloring/models"
    model_path = os.path.join(model_save_path, model_name)

    hostname = socket.gethostname()
    # Generate parameters based on the search method
    dataset = GraphColoringDataset(root=data_path, graph_type="gc_specific")            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    first_batch_iter = iter(loader)
    first_batch = next(first_batch_iter)
    metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
    num_hidden_channels = hidden_units
    input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
    hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
    out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
    model = GCGNN(metadata, input_size, out_channels, hidden_size, num_passes=num_lstm_passes, device=device)
    model = model.cuda()

    group = "gc_specific"
    
    wandb.init(
        project=f"GC-GNN",
        name=f"test_{model_name}"
        group=group
    )
    process_model(model, None, criterion, loader, mode='test', samples_per_epoch=None, model_path=None, clip_value=None)
    wandb.finish() 