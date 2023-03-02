import torch
from sat_parser import parse_dimacs_cnf
from model import SatGNN
from dataset import SatDataset
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss
import torch.nn.functional as F


# def train_sat_gnn(data_dir, model=SatGNN(hidden_channels=64)):
#     cnf = parse_dimacs_cnf(test_path)
#     data = cnf.build_heterogeneous_graph()
#     data = T.ToUndirected()(data)
#     model = to_hetero(model, data.metadata(), aggr='mean', debug=True)
#     model(data.x_dict, data.edge_index_dict, data.batch_dict)


if __name__ == "__main__":
    from torch_geometric.datasets import OGB_MAG
    
    test_path = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\data"
    
    dataset = SatDataset(root=test_path)
    train_dataset = dataset[:26]
    test_dataset = dataset[26:]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SatGNN(hidden_channels=64, num_layers=2)
    model = model.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
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
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(1, 171):
        train_acc = train()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')