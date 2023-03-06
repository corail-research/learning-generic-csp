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
    
    dataset = SatDataset(root=test_path)#.shuffle()
    train_dataset = dataset[100:]
    test_dataset = dataset[:100]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # model = SatGNN(hidden_channels=64, num_layers=2)
    model = HGT(64, 2, 2, 4, train_dataset[0])
    model = model.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    def train():
        model.train()
        right_classification = 0
        total_examples = 0
        total_loss = 0
        for data in train_loader:

            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            loss = criterion(out, data["variable"].y)
            loss.backward()
            optimizer.step()
            right_classification += sum(out.argmax(dim=1) == data["variable"].y.argmax(dim=1)).item()
            total_examples += len(data)
            total_loss +=  loss.item()
        
        return right_classification/total_examples, float(total_loss/total_examples)
    
    train()

    def test(loader):
        model.eval()
        right_classification = 0
        total_examples = 0
        total_loss = 0
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  
            loss = criterion(out, data["variable"].y)
            right_classification += sum(out.argmax(dim=1)
                                        == data["variable"].y.argmax(dim=1)).item()
            total_examples += len(data)
            total_loss += loss.item()

        return right_classification/total_examples, float(total_loss/total_examples)


    for epoch in range(1, 171):
        
        train_acc, train_loss = train()
        test_acc, test_loss = test(test_loader)

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')
        print("=====================================")