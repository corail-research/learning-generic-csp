import argparse
import os
import subprocess
import wandb
from parse import parse
from config import CONFIGS


parser = argparse.ArgumentParser()
parser.add_argument('--train', action=argparse.BooleanOptionalAction, dest="train")
parser.add_argument('--test', action=argparse.BooleanOptionalAction, dest="test")

parser.add_argument('--train_dir', action='store', dest='train_dir', type=str, default='data/train/sr5')
parser.add_argument('--test_dir', action='store', dest='test_dir', type=str, default='data/test/sr5')
parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=40)
parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=0)
parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=-1)
parser.add_argument('--n_rounds', action='store', dest='n_rounds', type=int, default=16)

args = parser.parse_args()

for config in CONFIGS:
    n_batches = 0
    for file in os.listdir(args.train_dir):
        parsed = parse("{}npb={}_nb={}.pkl",file)
        max_nodes_per_batch = parsed[1]
        n_batches += int(parsed[2])

    config['Max nodes per batch'] = int(max_nodes_per_batch)
    config['Number of batches'] = n_batches

    run = wandb.init(
        project='neurosat',
        entity='seapearl',
        config=config
    )
    wandb_run_id = run.id
    print(args)


    subprocess.call(
            f"python3 python/train.py {args.train_dir} --n_epochs {args.n_epochs} --run_id {args.run_id} --wandb_id {wandb_run_id} --test_dir {args.test_dir}",
            shell=True,
        )

wandb.finish()