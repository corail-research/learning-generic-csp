import socket
from datetime import datetime
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
from models.decision_tsp.base_model import GNNTSP
from models.decision_tsp.dataset import DTSPDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import train_model
from models.decision_tsp.config import TSPExperimentConfig
from models.common.pytorch_lr_scheduler import  GradualWarmupScheduler
from models.common.pytorch_samplers import  PairNodeSampler, PairBatchSampler

if __name__ == "__main__":
    import math
    search_method = "random"  # Set to either "grid" or "random"
    # data_path = r"./src/models/decision_tsp/data"
    data_path = r"/scratch1/boileo/dtsp/data"
    # Hyperparameters for grid search or random search
    batch_sizes = [32]
    hidden_units = [256]
    num_heads = [2]
    start_learning_rates = [0.00002]
    num_lstm_passes = [32]
    num_layers = [2]
    dropout = [0.1]
    num_epochs = 400
    device = "cuda:0"
    train_ratio = 0.99
    samples_per_epoch = 4096
    nodes_per_batch= [12000]
    use_sampler_loader = False
    weight_decay = [0.0000000001]
    num_epochs_lr_warmup = 5
    num_epochs_lr_decay = 20
    lr_decay_factor = 0.8
    generic_representation = False
    target_deviation = 0.02
    clip_gradient_norm = 0.65
    gnn_aggregation = "mean"
    
    hostname = socket.gethostname()

    experiment_config = TSPExperimentConfig(
        batch_sizes=batch_sizes,
        hidden_units=hidden_units,
        num_heads=num_heads,
        start_learning_rates=start_learning_rates,
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
        num_epochs_lr_warmup=num_epochs_lr_warmup,
        num_epochs_lr_decay=num_epochs_lr_decay,
        lr_decay_factor=lr_decay_factor,
        generic_representation=generic_representation,
        target_deviation=target_deviation,
        clip_gradient_norm=clip_gradient_norm,
        lr_scheduler_patience=13,
        lr_scheduler_factor=0.8,
        layernorm_lstm_cell=True
    )

    # Generate parameters based on the search method
    if search_method == "grid":
        search_parameters = experiment_config.generate_grid_search_parameters()
    elif search_method == "random":
        num_random_combinations = 10
        search_parameters = experiment_config.generate_random_search_parameters(num_random_combinations)
    else:
        raise ValueError("Invalid search_method. Must be 'grid' or 'random'")
    if experiment_config.generic_representation:
        dataset = DTSPDataset(root=experiment_config.data_path, graph_type="generic", meta_connected_to_all=False, in_memory=False)
    else:
        dataset = DTSPDataset(root=experiment_config.data_path, graph_type="dtsp_specific", meta_connected_to_all=False, in_memory=False)
    train_dataset = dataset[:math.floor(len(dataset) * experiment_config.train_ratio)]
    test_dataset = dataset[math.floor(len(dataset) * experiment_config.train_ratio):]
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    for params in search_parameters:
        if params.use_sampler_loader:
            train_sampler = PairNodeSampler(train_dataset, params.nodes_per_batch)
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        else:
            train_sampler = PairBatchSampler(train_dataset, params.batch_size) 
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
        first_batch_iter = iter(test_loader)
        first_batch = next(first_batch_iter)
        metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
        num_hidden_channels = params.hidden_units
        input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
        hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
        out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
        model = GNNTSP(metadata, input_size, out_channels, hidden_size, num_passes=params.num_lstm_passes, device=device, aggr=params.gnn_aggregation)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=params.start_learning_rate, weight_decay=params.weight_decay)
        if type(model) == GNNTSP:
            group = f"dtsp_specific_dev={target_deviation}" + "_" + hostname
        else:
            group = f"generic_dev={target_deviation}"+ "_" + hostname
        wandb.init(
            project=f"TSP-GNN",
            config=params,
            group=group
        )
        if params.lr_scheduler_patience is not None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.lr_scheduler_factor, patience=params.lr_scheduler_patience, verbose=True)
        else:
            lr_scheduler = None
        # train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, warmup_scheduler, criterion, params["num_epochs"], samples_per_epoch=samples_per_epoch, clip_value=experiment_config.clip_gradient_norm)
        profile = cProfile.Profile()
        profile.run('train_model(model, train_loader, test_loader, optimizer, lr_scheduler, criterion, params.num_epochs, samples_per_epoch=params.samples_per_epoch, clip_value=experiment_config.clip_gradient_norm)')

        stats = pstats.Stats(profile)
        stats.sort_stats('tottime')
        stats.print_stats()
        wandb.finish() 