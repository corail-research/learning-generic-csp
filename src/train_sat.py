import socket
from datetime import datetime
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
from models.common.lstm_conv import AdaptedNeuroSAT
from models.sat.neurosat_model import NeuroSAT
from models.sat.dataset import SatDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from models.common.training_utils import train_model
from models.sat.config import SATExperimentConfig
from models.common.pytorch_lr_scheduler import  GradualWarmupScheduler
from models.common.pytorch_samplers import  PairNodeSampler, PairBatchSampler

if __name__ == "__main__":
    import math
    search_method = "grid"  # Set to either "grid" or "random"
    # data_path = r"./src/models/sat/generic/temp_remote_date" # local
    data_path = r"/scratch1/boileo/sat/data/sat_specific" # servercd 
    # Hyperparameters for grid search or random search
    batch_sizes = [2]
    hidden_units = [128, 256]
    start_learning_rates = [0.00002]
    num_lstm_passes = [26]
    num_layers = [3]
    dropout = [0.1]
    num_epochs = 400
    device = "cuda:0"
    train_ratio = 0.8
    samples_per_epoch = 200000
    nodes_per_batch= [12000]
    use_sampler_loader = False
    weight_decay = [0.0000000001]
    lr_decay_factor = 0.8
    generic_representation = True
    gnn_aggregation = "add"
    model_save_path = "./scratch1/boileo/sat/models"
    # model_save_path = r"./src/models/sat/models/"
    
    hostname = socket.gethostname()

    experiment_config = SATExperimentConfig(
        batch_sizes=batch_sizes,
        hidden_units=hidden_units,
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
        lr_decay_factor=lr_decay_factor,
        generic_representation=generic_representation,
        flip_inputs=False,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.2,
        layernorm_lstm_cell=True,
        gnn_aggregation=gnn_aggregation,
        model_save_path=model_save_path
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
        dataset = SatDataset(root=experiment_config.data_path, graph_type="generic", meta_connected_to_all=False, in_memory=False)
    else:
        dataset = SatDataset(root=experiment_config.data_path, graph_type="sat_specific", meta_connected_to_all=False, in_memory=False)
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
        # model_type = "sat_spec"
        model = NeuroSAT(
            metadata,
            input_size,
            out_channels,
            hidden_size,
            num_passes=params.num_lstm_passes,
            device=device,
            flip_inputs=params.flip_inputs,
            layernorm_lstm_cell=params.layernorm_lstm_cell,
            aggr=params.gnn_aggregation
        )
        # model = AdaptedNeuroSAT(metadata, input_size, out_channels, hidden_size, num_passes=params.num_lstm_passes, device=device)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=params.start_learning_rate, weight_decay=params.weight_decay)
        if type(model) == AdaptedNeuroSAT:
            group = "generic" + hostname
        else:
            group = "sat_specific" + hostname
        wandb.init(
            project=f"SATGNN",
            config=params,
            group=group
        )
        if params.lr_scheduler_patience is not None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.lr_scheduler_factor, patience=params.lr_scheduler_patience, verbose=True)
        else:
            lr_scheduler = None
        train_model(
            model,
            train_loader,
            test_loader,
            optimizer,
            lr_scheduler,
            criterion,
            params.num_epochs,
            samples_per_epoch=params.samples_per_epoch,
            clip_value=params.clip_gradient_norm,
            model_save_path=params.model_save_path,
            wandb_run_name=wandb.run.name
        )
        # profile = cProfile.Profile()
        # profile.run('train_model(model, train_loader, test_loader, optimizer, lr_scheduler, criterion, params.num_epochs, samples_per_epoch=params.samples_per_epoch, clip_value=0.65)')

        # stats = pstats.Stats(profile)
        # stats.sort_stats('tottime')
        # stats.print_stats()
        wandb.finish() 