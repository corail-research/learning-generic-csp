import argparse
import wandb
import torch
import os
from sklearn.metrics import classification_report
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

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, num_epochs, samples_per_epoch=None, threshold=0.6):
    train_losses, test_losses, train_metrics, test_metrics = [], [], [], []

    for epoch in range(1, num_epochs):
        scheduler.step(epoch)
        train_acc, train_loss, train_metric = process_model(model, optimizer, criterion, train_loader, mode='train', samples_per_epoch=samples_per_epoch)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        test_acc, test_loss, test_metric = process_model(model, None, criterion, test_loader, mode='test', samples_per_epoch=samples_per_epoch)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)

        print(f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    return train_losses, test_losses, train_metrics, test_metrics

def process_model(model, optimizer, criterion, loader, mode='train', samples_per_epoch=None, model_path=None):
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'."

    if mode == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    y_true, y_pred = [], []
    
    num_samples = 0
    for data in loader:
        data = data.to(device="cuda:0")
        if mode == 'train':
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            if out.size(1) == 1:
                out = out.squeeze(1)
            loss = criterion(out, data["variable"].y.float())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
                if out.size(1) == 1:
                    out = out.squeeze(1)
                loss = criterion(out, data["variable"].y.float())
        
        predicted = (out > 0).int()
        label = data["variable"].y.int()

        y_true.extend(label.tolist())
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