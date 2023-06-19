import argparse
from datetime import datetime
import wandb
import torch
import numpy as np
import random
import os
from sklearn.metrics import classification_report
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

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, threshold=0.6):
    train_losses, test_losses, train_metrics, test_metrics = [], [], [], []

    for epoch in range(1, num_epochs):
        train_acc, train_loss, train_metric = process_model(model, optimizer, criterion, train_loader, mode='train')
        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        test_acc, test_loss, test_metric = process_model(model, None, criterion, test_loader, mode='test')
        test_losses.append(test_loss)
        test_metrics.append(test_metric)

        if epoch >= 100:
            last_10_avg_train = np.mean([tm['train/global_acc'] for tm in train_metrics[-10:]])
            last_10_avg_eval = np.mean([tm['test/global_acc'] for tm in test_metrics[-10:]])
            if last_10_avg_eval < threshold or last_10_avg_train < threshold:
                return None, None, None, None

        print(f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    return train_losses, test_losses, train_metrics, test_metrics

def process_model(model, optimizer, criterion, loader, mode='train'):
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'."

    if mode == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device="cuda:0")
        if mode == 'train':
            optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        loss = criterion(out, data["variable"].y)
        if mode == 'train':
            loss.backward()
            optimizer.step()

        predicted = out.argmax(dim=1).cpu()
        label = data["variable"].y.cpu()
        y_true.extend(label.tolist())
        y_pred.extend(predicted.tolist())

        total_loss += loss.item()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        f"{mode}/global_acc": report['accuracy'],
        f"{mode}/acc_class_0": report['0']['precision'],
        f"{mode}/acc_class_1": report['1']['precision'],
        f"{mode}/f1_class_0": report['0']['f1-score'],
        f"{mode}/f1_class_1": report['1']['f1-score'],
        f"{mode}/loss": total_loss / len(y_true)
    }
    wandb.log(metrics)

    return report['weighted avg']["precision"], total_loss / len(y_true), metrics


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
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\generic_sat\data\train"

    # Hyperparameters for grid search or random search
    batch_sizes = [32]
    hidden_units = [64]
    num_heads = [2, 4]
    learning_rates = [0.001, 0.0005]
    num_layers = [3]
    dropout = 0.3
    num_epochs = 200
    device = "cuda:0"

    dataset = SatDataset(root=test_path, graph_type="refactored", meta_connected_to_all=True, use_sat_label_as_feature=False)
    train_dataset = dataset[:280]
    test_dataset = dataset[280:]
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
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
        # model = GatedUpdate(params["num_hidden_units"], 2, params["num_heads"], params["num_layers"], train_dataset[0], dropout_prob=params["dropout"])
        model = GatedUpdate(params["num_hidden_units"], 2, params["num_heads"], params["num_layers"], train_dataset[0], dropout_prob=params["dropout"])
        model = model.to(device)
        # optimizer = torch.optim.Adam(model.param_list,lr=params["learning_rate"],weight_decay=0.0000000001)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)

        train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, criterion, params["num_epochs"])
        wandb.finish()
    
    