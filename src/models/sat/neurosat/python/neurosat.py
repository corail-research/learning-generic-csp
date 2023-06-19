# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import namedtuple
import numpy as np
import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import wandb
from confusion import ConfusionMatrix
from problems_loader import init_problems_loader
from mlp import MLP, LayerNormLSTMCell
from util import repeat_end, decode_final_reducer, decode_transfer_fn
from sklearn.cluster import KMeans



class NeuroSAT(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm_state_tuple = namedtuple('LSTMState',('h','c'))
        self.final_reducer = (lambda x: torch.mean(x, dim=[1, 2]))

        opts = self.opts

        self.L_init = torch.empty((1,opts.d),device=self.device)
        self.C_init = torch.empty((1,opts.d),device=self.device)
        nn.init.normal_(self.L_init)
        nn.init.normal_(self.C_init)


        self.LC_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d),device=self.device)
        self.CL_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d),device=self.device)

        self.L_update = LayerNormLSTMCell(opts, activation=decode_transfer_fn(opts.lstm_transfer_fn), state_tuple=self.lstm_state_tuple, device=self.device)
        self.C_update = LayerNormLSTMCell(opts, activation=decode_transfer_fn(opts.lstm_transfer_fn), state_tuple=self.lstm_state_tuple, device=self.device)

        self.L_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1),device=self.device)
    
        self.vote_bias = nn.Parameter(torch.zeros(1,device=self.device))
        self.train_problems_loader = None        
        self.learning_rate = opts.lr_start
        self.param_list = list(self.LC_msg.parameters()) + list(self.CL_msg.parameters()) \
            + list(self.L_vote.parameters()) + [self.vote_bias] \
            + list(self.L_update.parameters()) + list(self.C_update.parameters())
            

        self.optimizer = torch.optim.Adam(self.param_list,lr=self.learning_rate,weight_decay=opts.l2_weight)

        if opts.lr_decay_type == "no_decay":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=self.optimizer,
                factor=0,
                total_iters=0,
            )

        elif opts.lr_decay_type == "poly":
            raise OSError
            last_epoch = opts.lr_decay_steps + 1 - 1 /(1 - (opts.lr_start/opts.lr_end)**(1/(opts.lr_power * opts.lr_decay_steps)))

            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer=self.optimizer,
                power=opts.lr_power,
                total_iters=opts.lr_decay_steps,
            )

        elif opts.lr_decay_type == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=opts.lr_decay,
            )
        else:
            raise Exception("lr_decay_type must be 'no_decay', 'poly' or 'exp'")


        self.saver = ModelSaver(f"snapshots/run{opts.run_id}",max_to_keep=self.opts.n_saves_to_keep)

    def init_random_seeds(self):
        torch.manual_seed(self.opts.tf_seed)
        np.random.seed(self.opts.np_seed)


    def flip(self, lits):
        return torch.cat([lits[self.n_vars:(2*self.n_vars), :], lits[0:self.n_vars, :]], dim=0)

    def pass_messages(self):

        denom = torch.sqrt(torch.tensor(self.opts.d, device=self.device, dtype=torch.float))

        L_output = torch.tile(torch.div(self.L_init, denom), [self.n_lits, 1])
        C_output = torch.tile(torch.div(self.C_init, denom), [self.n_clauses, 1])

        L_state = self.lstm_state_tuple(h=L_output, c=torch.zeros([self.n_lits, self.opts.d], device=self.device))
        C_state = self.lstm_state_tuple(h=C_output, c=torch.zeros([self.n_clauses, self.opts.d], device=self.device))

        i = 0
        while i < self.opts.n_rounds:
            i += 1
            LC_pre_msgs = self.LC_msg.forward(L_state.h)
            LC_msgs = self.L_unpack.t() @ LC_pre_msgs

            C_state = self.C_update(inputs=LC_msgs, state=C_state)

            CL_pre_msgs = self.CL_msg.forward(C_state.h)
            CL_msgs = self.L_unpack @ CL_pre_msgs

            L_state = self.L_update(inputs=torch.cat([CL_msgs, self.flip(L_state.h)], axis=1), state=L_state)


        self.final_lits = L_state.h
        self.final_clauses = C_state.h

    def compute_logits(self):
        self.all_votes = self.L_vote.forward(self.final_lits) # n_lits x 1
        self.all_votes_join = torch.cat([self.all_votes[0:self.n_vars], self.all_votes[self.n_vars:self.n_lits]], axis=1) # n_vars x 2

        self.all_votes_batched = torch.reshape(self.all_votes_join, [self.n_batches, self.n_vars_per_batch, 2])
        self.logits = self.final_reducer(self.all_votes_batched) + self.vote_bias
        # print((self.all_votes > 0).count_nonzero()/self.all_votes.size()[0])
        # print(self.vote_bias)

    def compute_cost(self):
        
        self.predict_costs = F.binary_cross_entropy_with_logits(self.logits, self.is_sat.float())
        self.predict_cost = torch.mean(self.predict_costs)

        self.cost = self.predict_cost



    def save(self, epoch):
        self.saver.save(self,epoch)

    def restore(self):
        if self.opts.restore_epoch == -1:
            epoch_id = 'final'
        else:
            epoch_id = self.opts.restore_epoch
        snapshot = f"snapshots/run{self.opts.restore_id}/model_epoch_{epoch_id}"
        self.saver.restore(self, snapshot)

    def build_feed_dict(self, problem):
        d = {}
        d['n_vars'] = torch.tensor(problem.n_vars,device=self.device)
        d['n_lits'] = torch.tensor(problem.n_lits,device=self.device)
        d['n_clauses'] = torch.tensor(problem.n_clauses,device=self.device)
        d['L_unpack'] = torch.sparse_coo_tensor(indices=torch.tensor(problem.L_unpack_indices.T, dtype=torch.long),
                                           values=torch.tensor(np.ones(problem.L_unpack_indices.shape[0]), dtype=torch.float),
                                           size=(problem.n_lits, problem.n_clauses))
        d['is_sat'] = torch.tensor(problem.is_sat,device=self.device)
        return d

    def rollout(self, feed_dict, find_sol_data=False):


        self.n_vars = feed_dict['n_vars']
        self.n_lits = feed_dict['n_lits']
        self.n_clauses = feed_dict['n_clauses']
        self.L_unpack = feed_dict['L_unpack'].to(self.device)
        self.is_sat = feed_dict['is_sat']

        self.n_batches = self.is_sat.size()[0]
        self.n_vars_per_batch = self.n_vars // self.n_batches

        self.pass_messages()
        self.compute_logits()
        self.compute_cost()

        self.optimizer.zero_grad()
        self.cost.backward()
        nn.utils.clip_grad_norm_(self.param_list,max_norm=self.opts.clip_val)
        self.optimizer.step()

        if find_sol_data:
            return self.all_votes, self.final_lits, self.logits, self.cost

        return self.logits, self.cost
    
    def train_epoch(self, epoch):
        if self.train_problems_loader is None:
            self.train_problems_loader = init_problems_loader(self.opts.train_dir)

        epoch_start = time.perf_counter_ns()

        epoch_train_cost = 0.0
        epoch_train_mat = ConfusionMatrix()

        train_problems, train_filename = self.train_problems_loader.get_next()
        for problem in train_problems:
            d = self.build_feed_dict(problem)

            logits, cost = self.rollout(feed_dict=d)
            epoch_train_cost += cost
            epoch_train_mat.update(problem.is_sat, logits > 0)
        
        self.scheduler.step()

        epoch_train_cost /= len(train_problems)
        epoch_train_mat = epoch_train_mat.get_percentages()
        epoch_end = time.perf_counter_ns()

        learning_rate = self.scheduler.get_last_lr()
        self.save(epoch)

        return (train_filename, epoch_train_cost, epoch_train_mat, learning_rate, epoch_end - epoch_start)
    


    def test(self, test_data_dir):
        print(f"TESTING - {test_data_dir}")
        test_problems_loader = init_problems_loader(test_data_dir)
        results = []

        while test_problems_loader.has_next():
            test_problems, test_filename = test_problems_loader.get_next()

            epoch_test_cost = 0.0
            epoch_test_mat = ConfusionMatrix()

            for problem in test_problems:
                d = self.build_feed_dict(problem)
                logits, cost = self.rollout(feed_dict=d)
                epoch_test_cost += cost
                epoch_test_mat.update(problem.is_sat, logits > 0)
            
            epoch_test_cost /= len(test_problems)
            epoch_test_mat = epoch_test_mat.get_percentages()

            wandb.log({
                "Validation Accurarcy": epoch_test_mat.tt+ epoch_test_mat.ff,
                "Validation Unsat accuracy": epoch_test_mat.ff / (epoch_test_mat.ff + epoch_test_mat.ft),
                "Validation Sat accuracy": epoch_test_mat.tt / (epoch_test_mat.tt + epoch_test_mat.tf),
                "Validation Sat accuracy": epoch_test_mat.tt / (epoch_test_mat.tt + epoch_test_mat.tf),
                "Validation Sat f1 score": epoch_test_mat.tt / (epoch_test_mat.tt + 0.5*(epoch_test_mat.tf + epoch_test_mat.ft)),
                "Validation Unsat f1 score": epoch_test_mat.ff / (epoch_test_mat.ff + 0.5*(epoch_test_mat.tf + epoch_test_mat.ft)),
                "Validation Loss": epoch_test_cost
                }
            )

            results.append((test_filename, epoch_test_cost, epoch_test_mat))

        return results

    def find_solutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.rollout(feed_dict=d,find_sol_data=True)

        all_votes = all_votes.detach().cpu()
        final_lits = final_lits.detach().cpu()

        solutions = []
        for batch in range(len(problem.is_sat)):
            decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
            decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                print(xs)
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            if self.solves(problem, batch, decode_cheap_A): solutions.append(reify(decode_cheap_A))
            elif self.solves(problem, batch, decode_cheap_B): solutions.append(reify(decode_cheap_B))
            else:

                L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
                L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

                kmeans = KMeans(n_clusters=2, random_state=0,n_init='auto').fit(L)
                distances = kmeans.transform(L)
                scores = distances * distances

                def proj_vlit_flit(vlit):
                    if vlit < problem.n_vars: return vlit - batch * n_vars_per_batch
                    else:                     return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

                def decode_kmeans_A(vlit):
                    return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                        scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

                decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                if self.solves(problem, batch, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
                elif self.solves(problem, batch, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
                else: solutions.append(None)

        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                #print("[%d] %d" % (batch, vlit))
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True

    def init_lstm_size(self,size):

        self.C_update.set_input_size(size)
        self.L_update.set_input_size(size)

    


class ModelSaver:
    def __init__(self, save_dir, max_to_keep):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.saved_models = []

    def save(self, model, epoch):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)

        model_path = os.path.join(self.save_dir, f'model_epoch_final.pt')
        torch.save(model.state_dict(), model_path)

        self.saved_models.append(model_path)

        """ if len(self.saved_models) > self.max_to_keep:
            oldest_model = self.saved_models.pop(0)
            os.remove(oldest_model) """
        

    def restore(self, model:NeuroSAT, restore_epoch):
        if restore_epoch == -1:
            restore_epoch = 'final'

        model_path = f'{restore_epoch}.pt'

        if os.path.exists(model_path):
            model_state_dict: nn.Module = torch.load(model_path)

            for key in model_state_dict:

                if 'C_update.lstm_cell.weight_ih' in key:
                    model.C_update.set_input_size(model_state_dict[key].size()[-1])

                if 'L_update.lstm_cell.weight_ih' in key:
                    model.L_update.set_input_size(model_state_dict[key].size()[-1])



            model.load_state_dict(model_state_dict)
            print(f"Model restored from {model_path}")
        else:
            print(f"Model checkpoint not found: {model_path}")
            raise FileNotFoundError

