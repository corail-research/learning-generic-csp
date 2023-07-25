from datetime import datetime
import wandb
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile
import pstats
from neurosat_model import NeuroSAT
from dataset import SatDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


from utils import generate_grid_search_parameters, generate_random_search_parameters, train_model, PairSampler

if __name__ == "__main__":
    import math
    search_method = "random"  # Set to either "grid" or "random"
    data_path = r"../data/train_mid"
    # Hyperparameters for grid search or random search
    batch_sizes = [32]
    hidden_units = [128]
    num_heads = [2, 4]
    learning_rates = [0.000]
    num_lstm_passes = [26]
    num_layers = [2, 3]
    dropout = 0.1
    num_epochs = 100
    device = "cuda:0"
    train_ratio = 0.8
    samples_per_epoch = 4096
    
    # Generate parameters based on the search method
    if search_method == "grid":
        search_parameters = generate_grid_search_parameters(batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs, num_lstm_passes)
    elif search_method == "random":
        num_random_combinations = 10
        search_parameters = generate_random_search_parameters(num_random_combinations, batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs, num_lstm_passes)
    else:
        raise ValueError("Invalid search_method. Must be 'grid' or 'random'")
    
    dataset = SatDataset(root=data_path, graph_type="sat_specific", meta_connected_to_all=False)
    train_dataset = dataset[:math.floor(len(dataset) * train_ratio)]
    test_dataset = dataset[math.floor(len(dataset) * train_ratio):]
            
    # criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    date = str(datetime.now().date())

    for params in search_parameters:
        wandb.init(
            project=f"SATGNN",
            name=f'date={date}-bs={params["batch_size"]}-hi={params["num_hidden_units"]}-he{params["num_heads"]}-l={params["num_layers"]}-lr={params["learning_rate"]}-dr={dropout}-sat-spec',
            config=params
        )

        train_sampler = PairSampler(train_dataset, 12000)
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        # train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
        first_batch_iter = iter(test_loader)
        first_batch = next(first_batch_iter)
        metadata = (list(first_batch.x_dict.keys()), list(first_batch.edge_index_dict.keys()))

        num_hidden_channels = params["num_hidden_units"]
        input_size = {key: value.size(1) for key, value in first_batch.x_dict.items()}
        hidden_size = {key: num_hidden_channels for key, value in first_batch.x_dict.items()}
        out_channels = {key: num_hidden_channels for key in first_batch.x_dict.keys()}
        model = NeuroSAT(metadata, input_size, out_channels, hidden_size, num_passes=params["num_lstm_passes"], device=device, flip_inputs=True)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=params["learning_rate"],weight_decay=0.0000000001)
    
        # train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, criterion, params["num_epochs"], samples_per_epoch=samples_per_epoch)
        profile = cProfile.Profile()
        profile.run('train_model(model, train_loader, test_loader, optimizer, criterion, params["num_epochs"], samples_per_epoch=samples_per_epoch)')

        stats = pstats.Stats(profile)
        stats.sort_stats('tottime')
        stats.print_stats()
        wandb.finish()