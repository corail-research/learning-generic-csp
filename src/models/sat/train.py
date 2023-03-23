import argparse
from datetime import datetime
import wandb
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model import SatGNN, HGT, HGTMeta, HGTSATSpecific
from dataset import SatDataset
from torch_geometric.loader import DataLoader
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def get_args():
    parser = argparse.ArgumentParser(description='Your script description here.')
    parser.add_argument('-path', type=str, default='./data',help='Path to the data directory. Default: ./data')
    parser.add_argument('-lr', type=float, required=True,help='Learning rate for the optimizer.')
    parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training.')
    parser.add_argument('-num_layers', type=int, required=True, help='Number of transformer layers.')
    parser.add_argument('-num_heads', type=int, required=True, help='Number of attention heads in each transformer layer.')
    parser.add_argument('-num_epochs', type=int, default=100, help='Number of epochs to train. Default: 100')
    parser.add_argument('-hidden_size', type=int, required=True, help='Hidden size of the transformer layers.')
    parser.add_argument('-device', type=str, default='cuda:0', help='Device to use for training. Default: cuda:0')
    parser.add_argument('-train_bounds', nargs='+', type=int, required=True,help='Start and end indices of the training set.')
    parser.add_argument('-valid_bounds', nargs='+', type=int, required=True,help='Start and end indices of the validation set.')
    parser.add_argument('-dropout', type=float, default=0.0, help='Dropout probability. Default: 0.0')
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
            f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc}')
        print(
            f'Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc}')

    return train_losses, test_losses, train_accs, test_accs

def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    right_classification = 0
    total_examples = 0
    total_loss = 0
    for data in train_loader:
        data = data.to(device="cuda:0")
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        loss = criterion(out, data["variable"].y)
        loss.backward()
        optimizer.step()
        right_classification += sum(out.argmax(dim=1)
                                    == data["variable"].y.argmax(dim=1)).item()
        total_examples += len(data)
        total_loss += loss.item()
    
    train_acc, train_loss = right_classification/total_examples, float(total_loss)/total_examples

    train_metrics = {
        "train/global_acc": train_acc,
        "train/acc_sat": train_acc,
        "train/acc_unsat": train_acc,
        "train/loss": train_loss
    }
    wandb.log(train_metrics)

    return train_acc, train_loss

def test_model(model, loader, criterion):
    model.eval()
    right_classification = 0
    total_examples = 0
    total_loss = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device="cuda:0")
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            loss = criterion(out, data["variable"].y)
            right_classification += sum(out.argmax(dim=1)
                                        == data["variable"].y.argmax(dim=1)).item()
            total_examples += len(data)
            total_loss += loss.item()
    
    test_acc, test_loss = right_classification / total_examples, float(total_loss)/total_examples
    test_metrics = {
        "test/acc": test_acc,
        "test/loss": test_loss
    }
    wandb.log(test_metrics)

    return test_acc, test_loss

if __name__ == "__main__":
    test_path = r"./data"

    hidden_units = [64, 128]
    learning_rates = [0.001, 0.005]
    num_layers = [2, 3, 4]
    dropout = 0.3
    num_epochs = 200
    batch_sizes = [32]
    num_heads = [2, 4]
    device = "cuda:0"

    dataset = SatDataset(root=test_path, graph_type="modified", meta_connected_to_all=True)
    train_dataset = dataset[:1000]
    test_dataset = dataset[1000:1500]

    criterion = torch.nn.BCELoss(reduction="sum")
    date = str(datetime.now().date())
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for num_hidden_units in hidden_units:
            for heads in num_heads:
                for lr in learning_rates:
                    for layers in num_layers:
                        wandb.init(
                            project=f"generic-graph-rep-sat-{date}",
                            name=f"bs={batch_size}-hi={num_hidden_units}-he{heads}-l={layers}-lr={lr}-dr={dropout}-sat-spec",
                            config={
                                "batch_size": batch_size,
                                "epochs": num_epochs,
                                "batch_size": batch_size,
                                "num_layers": layers,
                                "learning_rate": lr,
                                "hidden_units": num_hidden_units,
                                "dropout": dropout,
                                "num_heads": heads
                            }
                        )
                        config = wandb.config

                        if dataset.graph_type == "refactored":
                            model = HGTMeta(num_hidden_units, 1, heads, layers, train_dataset[0], dropout_prob=dropout)
                        elif dataset.graph_type == "modified":
                            model = HGTSATSpecific(num_hidden_units, 2, heads, layers, train_dataset[0], dropout_prob=dropout)
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    
    wandb.finish()