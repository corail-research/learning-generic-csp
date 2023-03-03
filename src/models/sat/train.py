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



if __name__ == "__main__":
    from torch_geometric.datasets import OGB_MAG
    
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\data"
    
    dataset = SatDataset(root=test_path)
    train_dataset = dataset[:500]
    test_dataset = dataset[500:]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # model = SatGNN(hidden_channels=64, num_layers=2)
    model = HGT(64, 2, 2, 1, train_dataset[0])
    model = model.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        
        for data in train_loader:

            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            loss = criterion(out, data["variable"].y)
            loss.backward()
            optimizer.step()
        
        return float(loss)
    
    train()

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  
            loss = criterion(out, data["variable"].y)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            # correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return loss.item()


    for epoch in range(1, 171):
        train_acc = train()
        loss = test(test_loader)

        print(f'Epoch: {epoch:03d}, Train loss: {train_acc:.4f}, Test loss: {loss:.4f}')