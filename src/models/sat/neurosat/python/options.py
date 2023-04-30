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

import argparse

import wandb

from config import CONFIG

def add_neurosat_options(parser):
    parser.add_argument('--d', action='store', dest='d', type=int, default= CONFIG['d'])
    parser.add_argument('--n_rounds', action='store', dest='n_rounds', type=int, default= CONFIG['n_rounds'])

    parser.add_argument('--lr_decay_type', action='store', dest='lr_decay_type', type=str, default= CONFIG['lr_decay_type'])
    parser.add_argument('--lr_start', action='store', dest='lr_start', type=float, default= CONFIG['lr_start'])
    parser.add_argument('--lr_decay', action='store', dest='lr_decay', type=float, default= CONFIG['lr_decay'])
    parser.add_argument('--lr_decay_steps', action='store', dest='lr_decay_steps', type=float, default= CONFIG['lr_decay_steps'])
    parser.add_argument('--lr_power', action='store', dest='lr_power', type=float, default= CONFIG['lr_power'])

    parser.add_argument('--l2_weight', action='store', dest='l2_weight', type=float, default= CONFIG['l2_weight'])
    parser.add_argument('--clip_val', action='store', dest='clip_val', type=float, default= CONFIG['clip_val'])

    parser.add_argument('--lstm_transfer_fn', action='store', dest='lstm_transfer_fn', type=str, default= CONFIG['lstm_transfer_fn'])
    parser.add_argument('--vote_transfer_fn', action='store', dest='mlp_transfer_fn', type=str, default= CONFIG['vote_transfer_fn'])

    parser.add_argument('--final_reducer', action='store', dest='final_reducer', type=str, default= CONFIG['final_reducer'])

    parser.add_argument('--n_msg_layers', action='store', dest='n_msg_layers', type=int, default= CONFIG['n_msg_layers'])
    parser.add_argument('--n_vote_layers', action='store', dest='n_vote_layers', type=int, default= CONFIG['n_vote_layers'])

    parser.add_argument('--tf_seed', action='store', dest='tf_seed', type=int, default= CONFIG['tf_seed'])
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default= CONFIG['np_seed'])


    """ wandb.init(
        project="Neurosat",
        config=CONFIG,

    ) """
