import argparse
from datetime import datetime
import wandb
import torch
import numpy as np
import random
import os
from sklearn.metrics import classification_report
from torch.utils.data import Sampler
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, batches_per_epoch=None, threshold=0.6):
    train_losses, test_losses, train_metrics, test_metrics = [], [], [], []

    for epoch in range(1, num_epochs):
        train_acc, train_loss, train_metric = process_model(model, optimizer, criterion, train_loader, mode='train', batches_per_epoch=batches_per_epoch)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        test_acc, test_loss, test_metric = process_model(model, None, criterion, test_loader, mode='test', batches_per_epoch=batches_per_epoch)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)

        print(f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    return train_losses, test_losses, train_metrics, test_metrics

def process_model(model, optimizer, criterion, loader, mode='train', batches_per_epoch=None):
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'."

    if mode == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    y_true, y_pred = [], []
    
    batch_id = 0
    for data in loader:
        data = data.to(device="cuda:0")
        if mode == 'train':
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            loss = criterion(out, data["variable"].y.float())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
                loss = criterion(out, data["variable"].y.float())
        
        if out.size(1) == 1:
            predicted = (out > 0).int()
            label = data["variable"].y.int()
        else:
            predicted = out.argmax(dim=1).cpu()
            label = data["variable"].y.cpu()

        y_true.extend(label.tolist())
        y_pred.extend(predicted.tolist())

        total_loss += loss.item()
        batch_id += 1
        if mode == "train" and batches_per_epoch is not None and batch_id == batches_per_epoch:
            break

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        f"{mode}/global_acc": report['accuracy'],
        f"{mode}/acc_class_0": report['0']['recall'],
        f"{mode}/acc_class_1": report['1']['recall'],
        f"{mode}/loss": total_loss / len(y_true)
    }
    wandb.log(metrics)

    return report["accuracy"], total_loss / len(y_true), metrics


def generate_grid_search_parameters(batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs, num_lstm_passes):
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
                            "num_epochs": num_epochs,
                            "num_lstm_passes": random.choice(num_lstm_passes)
                        }

def generate_random_search_parameters(n, batch_sizes, hidden_units, num_heads, learning_rates, num_layers, dropout, num_epochs, num_lstm_passes):
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
            "num_epochs": num_epochs,
            "num_lstm_passes": random.choice(num_lstm_passes)
        }
        
        frozen_params = frozenset(params.items())
        
        if frozen_params not in tested_combinations:
            tested_combinations.add(frozen_params)
            count += 1
            yield params

class PairSampler(Sampler):
    def __init__(self, dataset, num_pairs):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.pair_list = self.compute_pair_list()

    def compute_pair_list(self):
        data1 = self.dataset[0]
        data2 = self.dataset[1]
        data3 = self.dataset[2]
        print(data1.filename[:9])
        print(data2.filename[:9])
        print(data3.filename[:9])

        if data1.filename[:9] == data2.filename[:9]:
            starting_point = 0
        elif data2.filename[:9] == data3.filename[:9]:
            starting_point = 1
        else:
            raise ValueError("None of the 3 first elements in the dataset form a pair \n Please check that the dataset is sorted by filename")
        
        # Create a list of all the pairs of data in the dataset
        pair_list = []
        for i in range(starting_point, len(self.dataset), 2):
            pair_list.append((i, i+1))

        return pair_list

    def __iter__(self):
        # Randomly sample pairs from the pair list
        selected_pairs = random.sample(self.pair_list, self.num_pairs)

        # Yield the indices of the selected pairs
        for pair in selected_pairs:
            yield pair[0]
            yield pair[1]

    def __len__(self):
        return self.num_pairs * 2
