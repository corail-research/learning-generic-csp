import torch
from torch import nn
from torch import optim
from problem_loader import init_problems_loader
from neurosat import NeuroSAT, train_epoch
from config_neurosat import NeuroSATConfig

train_dir = r"C:\Users\leobo\Desktop\École\Poly\Recherche\Generic-Graph-Representation\Graph-Representation\src\models\sat\data\raw"
config = NeuroSATConfig(
    d=64,
    n_msg_layers=2,
    n_vote_layers=2,
    n_rounds=26,
    l2_weight=0.001,
)

neurosat = NeuroSAT(config)

optimizer = optim.Adam(neurosat.parameters(), lr=config.lr_start)
train_problems_loader = init_problems_loader(train_dir)

n_epochs = 10  # Set the number of epochs

for epoch in range(n_epochs):
    train_filename, epoch_train_cost, epoch_train_mat = train_epoch(neurosat, train_problems_loader, optimizer, epoch)
    print(f"Epoch: {epoch+1}, Cost: {epoch_train_cost}, Train Accuracy: {epoch_train_mat}")

    # If you want to save the model after each epoch, you can do it here
    # torch.save(neurosat.state_dict(), f"neurosat_epoch_{epoch+1}.pt")

train_problems_loader.reset()  # Reset the problems loader after training
