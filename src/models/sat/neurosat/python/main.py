import random
import subprocess
import sys
import os
import argparse
from parse import parse
import wandb
from config import CONFIGS
# from options import add_neurosat_options
from neurosat import NeuroSAT


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', action='store', dest='train_dir', type=str, default='data/train/sr5')
parser.add_argument('--test_dir', action='store', dest='test_dir', type=str, default='data/test/sr5')
parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=1)
parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=0)
parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=-1)
parser.add_argument('--restore_id', action='store', dest='restore_id', type=int, default=None)
parser.add_argument('--n_rounds', action='store', dest='n_rounds', type=int, default=16)
parser.add_argument('--n_saves_to_keep', action='store', dest='n_saves_to_keep', type=int, default=4, help='Number of saved models to keep')
parser.add_argument('--wandb_id', action='store', dest='wandb_id', type=str, default=None, help='Number of saved models to keep')
opts = parser.parse_args()


def train_with_config(config):
    n_batches = 0
    for file in os.listdir(opts.train_dir):
        parsed = parse("{}npb={}_nb={}.pkl",file)
        max_nodes_per_batch = parsed[1]
        n_batches += int(parsed[2])

    config['Max nodes per batch'] = int(max_nodes_per_batch)
    config['Number of batches'] = n_batches

    setattr(opts, 'commit', subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())
    setattr(opts, 'hostname', subprocess.check_output(['hostname']).strip())

    if opts.run_id is None: opts.run_id = random.randrange(sys.maxsize)

    print(opts)

    if not os.path.exists("snapshots/"):
        os.mkdir("snapshots")

    g = NeuroSAT(config)
    
    run = wandb.init(
        project='neurosat',
        entity='seapearl',
        config=config
    )

    for epoch in range(opts.n_epochs):
        result = g.train_epoch(epoch)
        (efilename, etrain_cost, etrain_mat, lr, etime) = result
        
        wandb.log({
            "Training Accurracy": etrain_mat.tt+ etrain_mat.ff,
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
    wandb.finish()

if __name__ == '__main__':
    for config in CONFIGS:
        train_with_config(config)