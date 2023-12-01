import socket
from datetime import datetime
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
# from models.common.lstm_conv import AdaptedNeuroSAT
from generic_xcsp.generic_model import GenericModel
from generic_xcsp.dataset import XCSP3Dataset
from generic_xcsp.training_config import GenericExperimentConfig
from torch.utils.data import DataLoader
from torch_geometric.loader import DataListLoader 
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
# from torch_geometric.nn import DataParallel
from models.common.training_utils import train_model
from models.common.pytorch_samplers import  PairNodeSampler, PairBatchSampler, custom_hetero_collate_fn

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
    # data_path = r"./src/models/sat/generic/temp_remote_date" # local
    # data_path = r"/scratch1/boileo/knapsack/data"
    # data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\data"
    # Hyperparameters for grid search or random search
    batch_sizes = [128]
    hidden_units = [128, 256]
    start_learning_rates = [0.00002]
    num_lstm_passes = [28]
    num_layers = [3]
    dropout = [0.1]
    num_epochs = 1000
    device = "cuda"
    train_ratio = 0.8
    samples_per_epoch = 100000
    nodes_per_batch = [12000]
    use_sampler_loader = False
    weight_decay = [0.0000000001]
    lr_decay_factor = 0.5
    generic_representation = True
    gnn_aggregation = "add"
    model_save_path = None
    # project_name = "Generic-SAT"
    # project_name = "Generic-TSP"
    # project_name = "Generic-GC"
    project_name = "Test"

    if project_name == "Generic-TSP":
        data_path = r"/scratch1/boileo/dtsp/data/generic"
        model_save_path = r"/scratch1/boileo/dtsp/models"
    elif project_name == "Generic-GC":
        data_path = r"/scratch1/boileo/graph_coloring/data/generic"
        model_save_path = r"/scratch1/boileo/graph_coloring/models"
    elif project_name == "Generic-Knapsack":
        data_path = r"/scratch1/boileo/knapsack/data"
        model_save_path = r"/scratch1/boileo/knapsack/models"
    elif project_name == "Generic-SAT":
        data_path = r"/scratch1/boileo/sat/data/generic"
        model_save_path = r"/scratch1/boileo/sat/models"
    else:
        data_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\data"
        model_save_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\knapsack\models"
    
    hostname = socket.gethostname()

    experiment_config = GenericExperimentConfig(
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
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.2,
        layernorm_lstm_cell=True,
        gnn_aggregation=gnn_aggregation,
        model_save_path=model_save_path,
        optimal_deviation_difference=0.02
    )

    # Generate parameters based on the search method
    if search_method == "grid":
        search_parameters = experiment_config.generate_grid_search_parameters()
    elif search_method == "random":
        num_random_combinations = 10
        search_parameters = experiment_config.generate_random_search_parameters(num_random_combinations)
    else:
        raise ValueError("Invalid search_method. Must be 'grid' or 'random'")
    
    dataset = XCSP3Dataset(root=experiment_config.data_path, in_memory=False, target_deviation=0.02)
    limit_index = int(((len(dataset) * experiment_config.train_ratio) // 2) * 2)
    
    train_dataset = dataset[:limit_index]
    # flat_train_dataset = list(itertools.chain(*train_dataset))
    test_dataset = dataset[limit_index:]
    # flat_test_dataset = list(itertools.chain(*test_dataset))
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    for params in search_parameters:
        # if params.use_sampler_loader:
        #     train_sampler = PairNodeSampler(train_dataset, params.nodes_per_batch)
        #     train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        # else:
        #     train_sampler = PairBatchSampler(train_dataset, params.batch_size) 
        train_loader = DataLoader(train_dataset, batch_size=64//32, shuffle=True, collate_fn=custom_hetero_collate_fn, num_workers=0)
        
        
        test_loader = DataLoader(test_dataset, batch_size=min(1024, len(test_dataset)), shuffle=False, collate_fn=custom_hetero_collate_fn, num_workers=0)
        first_batch = train_dataset[0][0]
        metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
        num_hidden_channels = params.hidden_units
        input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
        hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
        out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
        # model_type = "sat_spec"
        model = GenericModel(
            metadata=metadata,
            in_channels=input_size,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_mlp_layers=params.num_layers,
            num_passes=params.num_lstm_passes,
            device=device,
            layernorm_lstm_cell=params.layernorm_lstm_cell,
            aggr=params.gnn_aggregation
        )
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=params.start_learning_rate, weight_decay=params.weight_decay)
        group = hostname
        wandb.init(
            project=project_name,
            config=params,
            group=group
        )
        if params.lr_scheduler_type == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.lr_scheduler_factor, patience=params.lr_scheduler_patience, verbose=True)
        elif params.lr_scheduler_type == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_scheduler_patience, gamma=params.lr_scheduler_factor, verbose=True)
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