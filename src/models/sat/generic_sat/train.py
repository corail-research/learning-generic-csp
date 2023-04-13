import argparse
from datetime import datetime
import wandb
import torch
import numpy as np
import random
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from model import SatGNN, HGT, HGTMeta, HGTSATSpecific, GatedUpdate
from dataset import SatDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def get_args():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("-path", type=str, default="./data",help="Path to the data directory. Default: ./data")
    parser.add_argument("-lr", type=float, required=True,help="Learning rate for the optimizer.")
    parser.add_argument("-batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("-num_layers", type=int, required=True, help="Number of transformer layers.")
    parser.add_argument("-num_heads", type=int, required=True, help="Number of attention heads in each transformer layer.")
    parser.add_argument("-num_epochs", type=int, default=100, help="Number of epochs to train. Default: 100")
    parser.add_argument("-hidden_size", type=int, required=True, help="Hidden size of the transformer layers.")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device to use for training. Default: cuda:0")
    parser.add_argument("-train_bounds", nargs="+", type=int, required=True,help="Start and end indices of the training set.")
    parser.add_argument("-valid_bounds", nargs="+", type=int, required=True,help="Start and end indices of the validation set.")
    parser.add_argument("-dropout", type=float, default=0.0, help="Dropout probability. Default: 0.0")
    args = parser.parse_args()
    
    return args

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, threshold=0.52):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(1, num_epochs):
        train_acc, train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_acc, test_loss = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if epoch >= 15:
            last_10_avg_train = np.mean(train_accs[-10:])
            last_10_avg_eval = np.mean(test_accs[-10:])
            if last_10_avg_eval < threshold or last_10_avg_train < threshold:
                return None, None, None, None

        print(
            f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(
            f"Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    return train_losses, test_losses, train_accs, test_accs


def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    total_sat_correct = 0
    total_unsat_correct = 0
    total_sat_examples = 0
    total_unsat_examples = 0

    for data in train_loader:
        data = data.to(device="cuda:0")
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict,
                    data.batch_dict, data.filename)
        loss = criterion(out, data["variable"].y)
        loss.backward()
        optimizer.step()

        # Count the number of correct predictions for SAT and UNSAT instances separately
        sat_correct = sum(out[data["variable"].y[:, 0] == 0].argmax(dim=1) == 0).item()
        unsat_correct = sum(
            out[data["variable"].y[:, 1] == 0].argmax(dim=1) == 1).item()
        total_sat_correct += sat_correct
        total_unsat_correct += unsat_correct

        # Count the number of SAT and UNSAT instances separately
        total_sat_examples += (data["variable"].y[:, 0] == 0).sum().item()
        total_unsat_examples += (data["variable"].y[:, 1] == 0).sum().item()

        total_loss += loss.item()

    # Calculate the accuracy on SAT and UNSAT instances separately
    sat_accuracy = total_sat_correct / total_sat_examples if total_sat_examples != 0 else 0
    unsat_accuracy = total_unsat_correct / total_unsat_examples if total_unsat_examples != 0 else 0

    # Calculate the overall accuracy
    total_examples = total_sat_examples + total_unsat_examples
    overall_accuracy = (total_sat_correct + total_unsat_correct) / total_examples if total_examples != 0 else 0

    train_metrics = {
        "train/global_acc": overall_accuracy,
        "train/acc_sat": sat_accuracy,
        "train/acc_unsat": unsat_accuracy,
        "train/loss": total_loss / total_examples
    }
    wandb.log(train_metrics)

    return overall_accuracy, total_loss / total_examples

def test_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_sat_correct = 0
    total_unsat_correct = 0
    total_sat_examples = 0
    total_unsat_examples = 0

    for data in loader:
        data = data.to(device="cuda:0")
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict, data.filename)
            loss = criterion(out, data["variable"].y)

            # Count the number of correct predictions for SAT and UNSAT instances separately
            sat_correct = sum(out[data["variable"].y[:, 0] == 0].argmax(dim=1) == 0).item()
            unsat_correct = sum(out[data["variable"].y[:, 1] == 0].argmax(dim=1) == 1).item()
            total_sat_correct += sat_correct
            total_unsat_correct += unsat_correct

            # Count the number of SAT and UNSAT instances separately
            total_sat_examples += (data["variable"].y[:, 0] == 0).sum().item()
            total_unsat_examples += (data["variable"].y[:, 1] == 0).sum().item()

            total_loss += loss.item()

    # Calculate the accuracy on SAT and UNSAT instances separately
    sat_accuracy = total_sat_correct / total_sat_examples if total_sat_examples != 0 else 0
    unsat_accuracy = total_unsat_correct / total_unsat_examples if total_unsat_examples != 0 else 0

    # Calculate the overall accuracy
    total_examples = total_sat_examples + total_unsat_examples
    overall_accuracy = (total_sat_correct + total_unsat_correct) / total_examples if total_examples != 0 else 0

    test_metrics = {
        "test/global_acc": overall_accuracy,
        "test/acc_sat": sat_accuracy,
        "test/acc_unsat": unsat_accuracy,
        "test/loss": total_loss / total_examples
    }
    wandb.log(test_metrics)

    return overall_accuracy, total_loss / total_examples

def generate_grid_search_parameters(batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs):
    for batch_size in batch_sizes:
        for num_hidden_units in hidden_units:
            for heads in num_heads:
                for lr in learning_rates:
                    for layers in num_layers:
                        yield {
                            "batch_size": batch_size,
                            "num_hidden_units": num_hidden_units,
                            "num_heads": heads,
                            "learning_rate": lr,
                            "num_layers": layers,
                            "dropout": dropout,
                            "num_epochs": num_epochs
                        }

def generate_random_search_parameters(n, batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs):
    tested_combinations = set()
    count = 0
    
    while count < n:
        params = {
            "batch_size": random.choice(batch_sizes),
            "num_hidden_units": random.choice(hidden_units),
            "num_heads": random.choice(num_heads),
            "learning_rate": random.choice(learning_rates),
            "num_layers": random.choice(num_layers),
            "dropout": dropout,
            "num_epochs": num_epochs
        }
        
        frozen_params = frozenset(params.items())
        
        if frozen_params not in tested_combinations:
            tested_combinations.add(frozen_params)
            count += 1
            yield params

if __name__ == "__main__":
    search_method = "grid"  # Set to either "grid" or "random"
    
    test_path = r"./data"
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\data"

    # Hyperparameters for grid search or random search
    batch_sizes = [32]
    hidden_units = [64, 128]
    num_heads = [2, 4]
    learning_rates = [0.001, 0.005]
    num_layers = [3, 4, 5]
    dropout = 0.3
    num_epochs = 200
    device = "cuda:0"

    dataset = SatDataset(root=test_path, graph_type="modified", meta_connected_to_all=True, use_sat_label_as_feature=False)
    train_dataset = dataset[:1000]
    test_dataset = dataset[1000:1500]
    criterion = torch.nn.BCELoss(reduction="sum")
    date = str(datetime.now().date())

    # Generate parameters based on the search method
    if search_method == "grid":
        search_parameters = generate_grid_search_parameters(batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs)
    elif search_method == "random":
        num_random_combinations = 10
        search_parameters = generate_random_search_parameters(num_random_combinations, batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs)
    else:
        raise ValueError("Invalid search_method. Must be 'grid' or 'random'")
    
    for params in search_parameters:
        wandb.init(
            project=f"generic-graph-rep-sat-{date}-logging-test",
            name=f'bs={params["batch_size"]}-hi={params["num_hidden_units"]}-he{params["num_heads"]}-l={params["num_layers"]}-lr={params["learning_rate"]}-dr={dropout}-sat-spec',
            config=params
        )
        
        # model = HGTMeta(params["num_hidden_units"], 2, params["num_heads"], params["num_layers"], train_dataset[0], dropout_prob=params["dropout"])
        model = GatedUpdate(params["num_hidden_units"], 2, params["num_heads"], params["num_layers"], train_dataset[0], dropout_prob=params["dropout"])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)

        train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, criterion, params["num_epochs"])
        wandb.finish()
    
    