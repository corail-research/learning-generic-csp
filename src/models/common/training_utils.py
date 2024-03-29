import argparse
import time
import wandb
import torch
import os
from sklearn.metrics import classification_report
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from models.decision_tsp.base_model import GNNTSP, GenericGNNTSP
from models.graph_coloring.base_model import GCGNN
from models.sat.neurosat_model import NeuroSAT
from src.models.knapsack.knapsack_model import KnapsackModel
import torch.nn.utils as utils

def train_model(model, train_loader, test_loader, optimizer, lr_scheduler, criterion, num_epochs, samples_per_epoch=None, clip_value=None, model_save_path=None, wandb_run_name=None):
    train_losses, test_losses, train_metrics, test_metrics = [], [], [], []
    best_acc = 0.0
    for epoch in range(1, num_epochs):
        start_time = time.time()
        train_acc, train_loss, train_metric = process_model(model, optimizer, criterion, train_loader, mode='train', samples_per_epoch=samples_per_epoch, clip_value=clip_value)
        train_time = time.time() - start_time

        start_time = time.time()
        test_acc, test_loss, test_metric = process_model(model, None, criterion, test_loader, mode='test', samples_per_epoch=samples_per_epoch)
        test_time = time.time() - start_time

        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)
        if lr_scheduler is not None:
            if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                lr_scheduler.step(test_loss)
            elif lr_scheduler.__class__.__name__ == "StepLR":
                lr_scheduler.step()

        print(f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Train time: {train_time:.2f} seconds")
        print(f"Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test time: {test_time:.2f} seconds")
        
        if test_acc > best_acc:
            best_acc = test_acc
            if model_save_path is not None:
                torch.save(model.state_dict(), os.path.join(model_save_path, f'{wandb_run_name}.pth'))

    return train_losses, test_losses, train_metric, test_metrics

def process_model(model, optimizer, criterion, loader, mode='train', samples_per_epoch=None, model_path=None, clip_value=None):
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'."

    if mode == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    y_true, y_pred = [], []
    
    num_samples = 0
    for data in loader:
        data = data.to(device="cuda")
        if type(model) == NeuroSAT:
            label = data["variable"].y.float()
        else:
            label = data.label.float()
        if mode == 'train':
            optimizer.zero_grad()
            if type(model) == KnapsackModel:
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict, data)
            else:
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            if out.size(1) == 1:
                out = out.squeeze(1)
            loss = criterion(out, label)
            loss.backward()
            if clip_value is not None:
                utils.clip_grad_norm_(model.parameters(), clip_value) 
            optimizer.step()
        else:
            with torch.no_grad():
                if type(model) == KnapsackModel:
                    out = model(data.x_dict, data.edge_index_dict, data.batch_dict, data)
                else:
                    out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
                if out.size(1) == 1:
                    out = out.squeeze(1)
                loss = criterion(out, label)
        
        predicted = (out > 0).int()
        y_true.extend(label.int().tolist())
        y_pred.extend(predicted.tolist())

        total_loss += loss.item()
        num_samples += len(label)
        if mode == "train" and samples_per_epoch is not None and num_samples >= samples_per_epoch:
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