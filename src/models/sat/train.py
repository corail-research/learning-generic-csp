import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from sat_parser import parse_dimacs_cnf
from model import SatGNN, HGT
from dataset import SatDataset
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


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
        right_classification += sum(out.argmax(dim=1) == data["variable"].y.argmax(dim=1)).item()
        total_examples += len(data)
        total_loss +=  loss.item()
    
    return right_classification/total_examples, float(total_loss/total_examples)

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

    return right_classification/total_examples, float(total_loss/total_examples)

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(1, num_epochs):
        train_acc, train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_acc, test_loss = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')
    
    return train_losses, test_losses, train_accs, test_accs
        
def plot_and_save(test_losses, train_losses, test_accs, train_accs, plot_name):
    fig, axs = plt.subplots(2)
    # Plotting the data for the first subplot
    axs[0].plot(range(len(train_losses)), train_losses, label="train loss")
    axs[0].set_title('Loss comparison')
    axs[0].plot(range(len(test_losses)), test_losses, label="test loss")
    axs[0].legend()

    axs[1].plot(range(len(train_accs)), train_accs, label="train acc")
    axs[1].set_title('Accuracy comparison')
    axs[1].plot(range(len(test_accs)), test_accs, label="test acc")
    axs[1].legend()

    # Adding labels and title to the overall plot
    fig.suptitle('Train vs test Comparison')
    fig.text(0.5, 0.04, 'Index', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')

    # Adjusting the layout and spacing
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.savefig(plot_name + ".png")

if __name__ == "__main__":
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\data"
    dataset = SatDataset(root=test_path)
    train_dataset = dataset[:5000]
    test_dataset = dataset[5000:5500]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    criterion = torch.nn.BCELoss()

    hidden_units = [128, 256]
    learning_rates = [0.001, 0.005, 0.1]
    num_layers = [5, 7]

    for num_hidden_units in hidden_units:
        for lr in learning_rates:
            for layers in num_layers:
                model = HGT(num_hidden_units, 2, 2, layers, train_dataset[0])
                model = model.to("cuda:0")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, optimizer, criterion, 300)
                plot_name = f"lr={lr}-num_layers={layers}-hidden_units={num_hidden_units}"
                plot_and_save(test_losses, train_losses, test_accs, train_accs, plot_name)

