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

import math
import numpy as np
import random
import datetime
import subprocess
import pickle
import sys
import os
import argparse

import wandb
from config import CONFIG
from options import add_neurosat_options
from neurosat import NeuroSAT

parser = argparse.ArgumentParser()
add_neurosat_options(parser)

parser.add_argument('train_dir', action='store', type=str, help='Directory with training data')
parser.add_argument('--test_dir', action='store', type=str, dest='test_dir',default=None)
parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=None)
parser.add_argument('--restore_id', action='store', dest='restore_id', type=int, default=None)
parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=None)
parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=100000, help='Number of epochs through data')
parser.add_argument('--n_saves_to_keep', action='store', dest='n_saves_to_keep', type=int, default=4, help='Number of saved models to keep')
parser.add_argument('--wandb_id', action='store', dest='wandb_id', type=str, default=None, help='Number of saved models to keep')

opts = parser.parse_args()

wandb.init(id=f"{opts.wandb_id}")

setattr(opts, 'commit', subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())
setattr(opts, 'hostname', subprocess.check_output(['hostname']).strip())

if opts.run_id is None: opts.run_id = random.randrange(sys.maxsize)

print(opts)

if not os.path.exists("snapshots/"):
    os.mkdir("snapshots")

g = NeuroSAT(opts)

config = CONFIG


for epoch in range(opts.n_epochs):
    result = g.train_epoch(epoch)
    (efilename, etrain_cost, etrain_mat, lr, etime) = result
    
    wandb.log({
        "Training Accurarcy": etrain_mat.tt+ etrain_mat.ff,
        "Training Unsat accuracy": etrain_mat.ff / (etrain_mat.ff + etrain_mat.ft),
        "Training Sat accuracy": etrain_mat.tt / (etrain_mat.tt + etrain_mat.tf),
        "Training Sat accuracy": etrain_mat.tt / (etrain_mat.tt + etrain_mat.tf),
        "Training Sat f1 score": etrain_mat.tt / (etrain_mat.tt + 0.5*(etrain_mat.tf + etrain_mat.ft)),
        "Training Unsat f1 score": etrain_mat.ff / (etrain_mat.ff + 0.5*(etrain_mat.tf + etrain_mat.ft)),
        "Training Loss": etrain_cost
        }
    )

    g.test(opts.test_dir)

    print("[Epoch %d] %.4f (%.2f, %.2f, %.2f, %.2f) ---> Accuracy: %.2f [%.2fs]" % (
        epoch,
        etrain_cost,
        etrain_mat.ff,
        etrain_mat.ft,
        etrain_mat.tf,
        etrain_mat.tt,
        etrain_mat.tt + etrain_mat.ff,
        etime/10**9))
