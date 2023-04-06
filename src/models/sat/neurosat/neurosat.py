import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import os
import time
from confusion import ConfusionMatrix
from problem_loader import init_problems_loader
from sklearn.cluster import KMeans

class MLP(nn.Module):
    def __init__(self, in_features, hidden_units, out_features, activation=nn.ReLU, dropout=0.5):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(in_features, hidden_units[0]))
        layers.append(activation())

        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            layers.append(activation())

        layers.append(nn.Linear(hidden_units[-1], out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class NeuroSAT(nn.Module):
    def __init__(self, config):
        super(NeuroSAT, self).__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the main components of NeuroSAT architecture
        self.L_init = nn.Parameter(torch.randn(1, config.d))
        self.C_init = nn.Parameter(torch.randn(1, config.d))

        self.LC_msg = MLP(config.d, [config.d] * config.n_msg_layers, config.d)
        self.CL_msg = MLP(config.d, [config.d] * config.n_msg_layers, config.d)

        self.L_update = nn.LSTMCell(config.d, config.d)
        self.C_update = nn.LSTMCell(config.d, config.d)

        self.L_vote = MLP(config.d, [config.d] * config.n_vote_layers, 1)
        self.vote_bias = nn.Parameter(torch.zeros(1))

        self.to(self.device)

    def forward(self, L_unpack, n_vars, n_lits, n_clauses, is_sat):
        n_batches = is_sat.size(0)
        n_vars_per_batch = n_vars // n_batches

        # Initialize literals and clauses
        L_state = (self.L_init.repeat(n_lits, 1), torch.zeros((n_lits, self.config.d), device=self.device))
        C_state = (self.C_init.repeat(n_clauses, 1), torch.zeros((n_clauses, self.config.d), device=self.device))

        # Pass messages
        for _ in range(self.config.n_rounds):
            LC_pre_msgs = self.LC_msg(L_state[0])
            LC_msgs = torch.sparse.mm(L_unpack, LC_pre_msgs)

            C_state = self.C_update(LC_msgs, C_state)

            CL_pre_msgs = self.CL_msg(C_state[0])
            CL_msgs = torch.sparse.mm(L_unpack, CL_pre_msgs)

            L_state = self.L_update(torch.cat([CL_msgs, torch.flip(L_state[0], [0])], dim=1), L_state)

        final_lits = L_state[0]
        final_clauses = C_state[0]

        all_votes = self.L_vote(final_lits)
        all_votes_join = torch.cat([all_votes[:n_vars], all_votes[n_vars:n_lits]], dim=1)

        all_votes_batched = all_votes_join.view(n_batches, n_vars_per_batch, 2)
        logits = torch.squeeze(torch.sum(all_votes_batched, dim=1)) + self.vote_bias

        return logits

    def compute_cost(self, logits, is_sat):
        predict_costs = nn.BCEWithLogitsLoss()(logits, is_sat.float())
        l2_cost = torch.zeros(1, device=self.device)

        for param in self.parameters():
            l2_cost += torch.norm(param, 2) ** 2

        cost = predict_costs + self.config.l2_weight * l2_cost
        return cost

def train_epoch(neurosat, train_problems_loader, optimizer, epoch):
    neurosat.train()

    epoch_train_cost = 0.0
    epoch_train_mat = ConfusionMatrix()

    train_problems, train_filename = train_problems_loader.get_next()
    for problem in train_problems:
        L_unpack = torch.sparse.FloatTensor(
            torch.LongTensor(problem.L_unpack_indices.T),
            torch.ones(problem.L_unpack_indices.shape[0]),
            torch.Size([problem.n_lits, problem.n_clauses])
        ).to(neurosat.device)

        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        is_sat = torch.tensor(problem.is_sat, dtype=torch.bool, device=neurosat.device)

        optimizer.zero_grad()

        logits = neurosat(L_unpack, n_vars, n_lits, n_clauses, is_sat)
        cost = neurosat.compute_cost(logits, is_sat)

        cost.backward()
        optimizer.step()

        epoch_train_cost += cost.item()
        epoch_train_mat.update(problem.is_sat, logits.cpu().detach().numpy() > 0)

    epoch_train_cost /= len(train_problems)
    epoch_train_mat = epoch_train_mat.get_percentages()

    return (train_filename, epoch_train_cost, epoch_train_mat)



