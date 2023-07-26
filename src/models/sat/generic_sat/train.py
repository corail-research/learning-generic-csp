from datetime import datetime
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
from model import AdaptedNeuroSAT
from neurosat_model import NeuroSAT
from dataset import SatDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from utils import train_model
from config import ExperimentConfig
from pytorch_utils import PairSampler, GradualWarmupScheduler

if __name__ == "__main__":
    import math
    search_method = "random"  # Set to either "grid" or "random"
    data_path = r"../sat_spec_data/train_mid"
    # Hyperparameters for grid search or random search
    batch_sizes = [32]
    hidden_units = [128]
    num_heads = [2, 4]
    learning_rates = [0.00002]
    num_lstm_passes = [26]
    num_layers = [2]
    dropout = [0.1]
    num_epochs = 2
    device = "cuda:0"
    train_ratio = 0.8
    samples_per_epoch = [4096]
    nodes_per_batch= [12000]
    use_sampler_loader = True
    weight_decay = [0.0000001]
    num_epochs_lr_warmup = 5
    num_epochs_lr_decay = 20
    lr_decay_factor = 0.8
    generic_representation = False
    
    experiment_config = ExperimentConfig(
        batch_sizes=batch_sizes,
        hidden_units=hidden_units,
        num_heads=num_heads,
        learning_rates=learning_rates,
        num_layers=num_layers,
        dropouts=dropout,
        num_epochs=num_epochs,
        num_lstm_passes=num_lstm_passes,
        device=device,
        train_ratio=train_ratio,
        samples_per_epochs=samples_per_epoch,
        nodes_per_batch=nodes_per_batch,
        data_path=data_path,
        use_sampler_loader=use_sampler_loader,
        weight_decay=weight_decay,
        num_epochs_lr_warmup=num_epochs_lr_warmup,
        num_epochs_lr_decay=num_epochs_lr_decay,
        lr_decay_factor=lr_decay_factor,
        generic_representation=generic_representation
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
        dataset = SatDataset(root=experiment_config.data_path, graph_type="generic", meta_connected_to_all=False)
    else:
        dataset = SatDataset(root=experiment_config.data_path, graph_type="sat_specific", meta_connected_to_all=False)
    train_dataset = dataset[:math.floor(len(dataset) * experiment_config.train_ratio)]
    test_dataset = dataset[math.floor(len(dataset) * experiment_config.train_ratio):]
            
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    for params in search_parameters:
        if params.use_sampler_loader:
            train_sampler = PairSampler(train_dataset, params.nodes_per_batch)
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=0)
        
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
        first_batch_iter = iter(test_loader)
        first_batch = next(first_batch_iter)
        metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))
        num_hidden_channels = params.hidden_units
        input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
        hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
        out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
        # model = NeuroSAT(metadata, input_size, out_channels, hidden_size, num_passes=params["num_lstm_passes"], device=device, flip_inputs=True)
        model = AdaptedNeuroSAT(metadata, input_size, out_channels, hidden_size, num_passes=params.num_lstm_passes, device=device)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=params.learning_rate, weight_decay=params.weight_decay)
        after_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: experiment_config.lr_decay_factor ** (epoch // experiment_config.num_epochs_lr_decay))
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=experiment_config.num_epochs_lr_warmup, after_scheduler=after_scheduler)

        if type(model) == AdaptedNeuroSAT:
            group = "generic"
        else:
            group = "sat_specific"
        wandb.init(
            project=f"SATGNN",
            config=params,
            group=group
        )
        # train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, warmup_scheduler, criterion, params["num_epochs"], samples_per_epoch=samples_per_epoch)
        profile = cProfile.Profile()
        profile.run('train_model(model, train_loader, test_loader, optimizer, warmup_scheduler, criterion, params.num_epochs, samples_per_epoch=params.samples_per_epoch)')

        stats = pstats.Stats(profile)
        stats.sort_stats('tottime')
        stats.print_stats()
        wandb.finish() 